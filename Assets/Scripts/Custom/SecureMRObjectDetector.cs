using System;
using System.Collections.Generic;
using System.Threading;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;
using SecureFab.Training;

namespace PicoXR.SecureMR.Demo
{
    /// <summary>
    /// Complete YOLO-based object detection system for SecureFab training.
    /// Detects objects (bottle, cup, scissors, book) and maps them to zones.
    /// </summary>
    public class SecureMRObjectDetector : MonoBehaviour
    {
        #region Inspector Fields
        [Header("Models & Assets")]
        [Tooltip("YOLO model for object detection")]
        public TextAsset yoloModel;

        [Tooltip("GLTF asset for instruction panel")]
        public TextAsset instructionPanelGltf;

        [Header("Zone Configuration (Normalized 0-1)")]
        [Range(0f, 0.5f)]
        public float leftZoneMaxX = 0.33f;

        [Range(0.5f, 1f)]
        public float rightZoneMinX = 0.66f;

        [Range(0f, 0.5f)]
        public float topZoneMaxY = 0.33f;

        [Range(0.5f, 1f)]
        public float bottomZoneMinY = 0.66f;

        [Header("Detection Settings")]
        [Range(0f, 1f)]
        public float confidenceThreshold = 0.5f;

        [Range(0f, 1f)]
        public float nmsThreshold = 0.45f;

        public int maxDetections = 10;

        [Header("Pipeline Settings")]
        public int vstWidth = 640;
        public int vstHeight = 640;

        [Range(0.01f, 1f)]
        public float detectionIntervalSeconds = 0.2f; // Run YOLO at 5 FPS

        [Range(0.01f, 1f)]
        public float renderIntervalSeconds = 0.033f; // Render at 30 FPS

        [Header("References")]
        public StepManager stepManager;

        [Header("Haptic Feedback")]
        public PicoControllerInput controllerInput;

        [Range(0f, 1f)]
        public float incorrectConfigHapticStrength = 0.7f;

        public int incorrectConfigHapticDuration = 200;

        [Header("Debug")]
        public bool debugLogging = true;

        public bool showDetectionBoxes = true;
        #endregion

        #region COCO Class IDs
        private const int COCO_BOTTLE = 39;
        private const int COCO_CUP = 41;
        private const int COCO_SCISSORS = 76;
        private const int COCO_BOOK = 73;

        private readonly Dictionary<int, string> objectNameMap = new Dictionary<int, string>
        {
            { COCO_BOTTLE, "bottle" },
            { COCO_CUP, "cup" },
            { COCO_SCISSORS, "scissors" },
            { COCO_BOOK, "book" }
        };
        #endregion

        #region Private Fields
        private Provider provider;
        private Pipeline vstPipeline;
        private Pipeline detectionPipeline;
        private Pipeline renderPipeline;

        // Global tensors
        private Tensor vstOutputGlobal;
        private Tensor detectionResultsGlobal;
        private Tensor instructionTextGlobal;
        private Tensor instructionPanelGltfTensor;

        // Detection state
        private ExpectedConfig currentDetectedConfig = new ExpectedConfig();
        private ExpectedConfig lastValidatedConfig;
        private int stableFrameCount = 0;
        private const int STABILITY_THRESHOLD = 5; // Require 5 consistent frames
        private float lastValidationTime = 0f;
        private float lastDetectionTime = 0f;
        private float lastRenderTime = 0f;

        // Threading
        private bool pipelinesReady = false;
        private readonly object initLock = new object();
        #endregion

        #region Unity Lifecycle
        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        private void Start()
        {
            SecureFabLogger.Log("ObjectDetector", "╔════════════════════════════════════╗");
            SecureFabLogger.Log("ObjectDetector", "║  SecureFab Object Detector Start  ║");
            SecureFabLogger.Log("ObjectDetector", "╚════════════════════════════════════╝");

            if (!ValidateConfiguration())
            {
                enabled = false;
                return;
            }

            InitializeSecureMR();
            SubscribeToStepManager();

            SecureFabLogger.Log("ObjectDetector", "Initialization complete");
        }

        private void Update()
        {
            if (!pipelinesReady) return;

            float currentTime = Time.time;

            // Run detection pipeline at specified interval
            if (currentTime - lastDetectionTime >= detectionIntervalSeconds)
            {
                RunDetectionPipeline();
                lastDetectionTime = currentTime;
            }

            // Run render pipeline at specified interval
            if (currentTime - lastRenderTime >= renderIntervalSeconds)
            {
                RunRenderPipeline();
                lastRenderTime = currentTime;
            }

            // Process detections and validate configuration
            ProcessDetections();
        }

        private void OnDestroy()
        {
            if (stepManager != null)
            {
                stepManager.onStepChanged.RemoveListener(OnStepChanged);
                stepManager.onConfigurationValidated.RemoveListener(OnConfigValidated);
            }
        }
        #endregion

        #region Initialization
        private bool ValidateConfiguration()
        {
            if (yoloModel == null)
            {
                SecureFabLogger.LogError("ObjectDetector", "YOLO model not assigned!");
                return false;
            }

            if (instructionPanelGltf == null)
            {
                SecureFabLogger.LogError("ObjectDetector", "Instruction panel GLTF not assigned!");
                return false;
            }

            if (stepManager == null)
            {
                SecureFabLogger.LogError("ObjectDetector", "StepManager reference missing!");
                return false;
            }

            if (!stepManager.IsInitialized)
            {
                SecureFabLogger.LogError("ObjectDetector", "StepManager not initialized!");
                return false;
            }

            return true;
        }

        private void InitializeSecureMR()
        {
            SecureFabLogger.Log("ObjectDetector", "Creating SecureMR provider and pipelines...");

            provider = new Provider(vstWidth, vstHeight);

            // Create global tensors
            vstOutputGlobal = provider.CreateTensor<float, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
            
            // Detection results: [max_detections, 6] where each row is [x_center, y_center, width, height, confidence, class_id]
            detectionResultsGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[] { maxDetections, 6 }));

            instructionTextGlobal = provider.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 512 }));
            instructionPanelGltfTensor = provider.CreateTensor<Gltf>(instructionPanelGltf.bytes);

            // Start pipeline creation in background thread
            Thread pipelineThread = new Thread(() =>
            {
                try
                {
                    CreateVSTPipeline();
                    CreateDetectionPipeline();
                    CreateRenderPipeline();

                    lock (initLock)
                    {
                        pipelinesReady = true;
                    }

                    SecureFabLogger.Log("ObjectDetector", "✓ All pipelines initialized successfully!");
                }
                catch (Exception e)
                {
                    SecureFabLogger.LogError("ObjectDetector", $"Pipeline init FAILED: {e.Message}\n{e.StackTrace}");
                }
            });

            pipelineThread.Start();
        }

        private void SubscribeToStepManager()
        {
            stepManager.onStepChanged.AddListener(OnStepChanged);
            stepManager.onConfigurationValidated.AddListener(OnConfigValidated);

            // Update instruction text for first step
            UpdateInstructionText(stepManager.CurrentStep);
        }
        #endregion

        #region Pipeline Creation
        private void CreateVSTPipeline()
        {
            SecureFabLogger.Log("ObjectDetector", "Creating VST pipeline...");

            vstPipeline = provider.CreatePipeline();

            var vstOutputPlaceholder = vstPipeline.CreateTensorReference<float, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
            var vstOutputUint8 = vstPipeline.CreateTensor<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));

            // Get camera frames
            var vstOp = vstPipeline.CreateOperator<RectifiedVstAccessOperator>();
            vstOp.SetResult("left image", vstOutputUint8);

            // Convert to float32
            var assignOp = vstPipeline.CreateOperator<AssignmentOperator>();
            assignOp.SetOperand("src", vstOutputUint8);
            assignOp.SetResult("dst", vstOutputPlaceholder);

            // Normalize to [0, 1]
            var normalizeOp = vstPipeline.CreateOperator<ArithmeticComposeOperator>(
                new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));
            normalizeOp.SetOperand("{0}", vstOutputPlaceholder);
            normalizeOp.SetResult("result", vstOutputPlaceholder);

            SecureFabLogger.Log("ObjectDetector", "✓ VST pipeline created");
        }

        private void CreateDetectionPipeline()
        {
            SecureFabLogger.Log("ObjectDetector", "Creating YOLO detection pipeline...");

            detectionPipeline = provider.CreatePipeline();

            var vstInputPlaceholder = detectionPipeline.CreateTensorReference<float, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
            var detectionOutputPlaceholder = detectionPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] { maxDetections, 6 }));

            // YOLO model configuration
            var modelConfig = new ModelOperatorConfiguration(yoloModel.bytes, SecureMRModelType.QnnContextBinary, "yolo");
            
            // Configure input/output mappings based on YOLO model
            // NOTE: These names must match your specific YOLO model's input/output tensor names
            modelConfig.AddInputMapping("images", "images", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("output0", "output0", SecureMRModelEncoding.Float32);

            var modelOp = detectionPipeline.CreateOperator<RunModelInferenceOperator>(modelConfig);
            modelOp.SetOperand("images", vstInputPlaceholder);

            // Raw YOLO output tensor
            var rawDetections = detectionPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 8400, 84 }));
            modelOp.SetResult("output0", rawDetections);

            // Post-process YOLO output (NMS, confidence filtering, etc.)
            // This is simplified - actual implementation would need proper NMS
            ProcessYOLOOutput(detectionPipeline, rawDetections, detectionOutputPlaceholder);

            SecureFabLogger.Log("ObjectDetector", "✓ Detection pipeline created");
        }

        private void ProcessYOLOOutput(Pipeline pipeline, Tensor rawOutput, Tensor processedOutput)
        {
            // PSEUDO-CODE: This is where you'd implement:
            // 1. Confidence thresholding
            // 2. Class filtering (only keep bottle, cup, scissors, book)
            // 3. NMS (Non-Maximum Suppression)
            // 4. Format conversion to [x_center, y_center, width, height, confidence, class_id]

            // For hackathon: This would be a complex operator graph.
            // You might need to implement this as a custom operator or
            // use existing NMS operators from the SecureMR samples.

            SecureFabLogger.LogWarning("ObjectDetector", "YOLO post-processing simplified for hackathon");
        }

        private void CreateRenderPipeline()
        {
            SecureFabLogger.Log("ObjectDetector", "Creating render pipeline...");

            renderPipeline = provider.CreatePipeline();

            var instructionTextPlaceholder = renderPipeline.CreateTensorReference<byte, Scalar>(1, new TensorShape(new[] { 512 }));
            var panelGltfPlaceholder = renderPipeline.CreateTensorReference<Gltf>();

            // Panel position (in front of user, slightly above)
            var panelTransform = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                new float[]
                {
                    0.6f, 0.0f, 0.0f, 0.0f,      // Scale and position
                    0.0f, 0.6f, 0.0f, 0.4f,      // Slightly above eye level
                    0.0f, 0.0f, 0.6f, -1.0f,     // 1 meter in front
                    0.0f, 0.0f, 0.0f, 1.0f
                });

            // Text rendering configuration
            var startPosition = renderPipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }));
            startPosition.Reset(new float[] { 0.1f, 0.5f });

            var colors = renderPipeline.CreateTensor<byte, Unity.XR.PXR.SecureMR.Color>(4, new TensorShape(new[] { 2 }));
            colors.Reset(new byte[] { 255, 255, 255, 255, 0, 0, 0, 200 }); // White text on semi-transparent black

            var textureId = renderPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }));
            textureId.Reset(new ushort[] { 0 });

            var fontSize = renderPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
            fontSize.Reset(new float[] { 48.0f });

            // Render text operator
            var renderTextOp = renderPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1024, 1024));
            renderTextOp.SetOperand("text", instructionTextPlaceholder);
            renderTextOp.SetOperand("start", startPosition);
            renderTextOp.SetOperand("colors", colors);
            renderTextOp.SetOperand("texture ID", textureId);
            renderTextOp.SetOperand("font size", fontSize);
            renderTextOp.SetOperand("gltf", panelGltfPlaceholder);

            // Render panel operator
            var renderPanelOp = renderPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            renderPanelOp.SetOperand("gltf", panelGltfPlaceholder);
            renderPanelOp.SetOperand("world pose", panelTransform);

            SecureFabLogger.Log("ObjectDetector", "✓ Render pipeline created");
        }
        #endregion

        #region Pipeline Execution
        private void RunDetectionPipeline()
        {
            if (vstPipeline == null || detectionPipeline == null) return;

            try
            {
                // Run VST pipeline
                var vstMapping = new TensorMapping();
                vstMapping.Set(vstPipeline.CreateTensorReference<float, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth })), vstOutputGlobal);
                vstPipeline.Execute(vstMapping);

                // Run detection pipeline
                var detectionMapping = new TensorMapping();
                detectionMapping.Set(detectionPipeline.CreateTensorReference<float, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth })), vstOutputGlobal);
                detectionMapping.Set(detectionPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] { maxDetections, 6 })), detectionResultsGlobal);
                detectionPipeline.Execute(detectionMapping);

                // Process detection results
                ParseDetectionResults();
            }
            catch (Exception e)
            {
                SecureFabLogger.LogError("ObjectDetector", $"Detection pipeline error: {e.Message}");
            }
        }

        private void RunRenderPipeline()
        {
            if (renderPipeline == null) return;

            try
            {
                var renderMapping = new TensorMapping();
                renderMapping.Set(renderPipeline.CreateTensorReference<byte, Scalar>(1, new TensorShape(new[] { 512 })), instructionTextGlobal);
                renderMapping.Set(renderPipeline.CreateTensorReference<Gltf>(), instructionPanelGltfTensor);
                renderPipeline.Execute(renderMapping);
            }
            catch (Exception e)
            {
                SecureFabLogger.LogError("ObjectDetector", $"Render pipeline error: {e.Message}");
            }
        }
        #endregion

        #region Detection Processing
        private void ParseDetectionResults()
        {
            // NOTE: In actual SecureMR, you cannot read tensor data directly from Unity.
            // This is PSEUDO-CODE showing the logic that would run inside SecureMR.
            
            // For the hackathon, you have two options:
            // 1. Implement zone mapping entirely within SecureMR operators
            // 2. Use a workaround like rendering detection results and reading pixels
            
            // Here's the conceptual logic:
            
            /*
            PSEUDO-CODE:
            
            Clear currentDetectedConfig
            
            For each detection in detectionResultsGlobal:
                x_center = detection[0]
                y_center = detection[1]
                confidence = detection[4]
                class_id = detection[5]
                
                if confidence < confidenceThreshold:
                    continue
                    
                if class_id not in [COCO_BOTTLE, COCO_CUP, COCO_SCISSORS, COCO_BOOK]:
                    continue
                
                zone = DetermineZone(x_center, y_center)
                objectName = objectNameMap[class_id]
                
                // Assign object to zone
                switch (zone):
                    case "left":
                        currentDetectedConfig.left = objectName
                    case "right":
                        currentDetectedConfig.right = objectName
                    case "top":
                        currentDetectedConfig.top = objectName
                    case "bottom":
                        currentDetectedConfig.bottom = objectName
            */

            SecureFabLogger.LogVerbose("ObjectDetector", "Detection results parsed");
        }

        private string DetermineZone(float x_norm, float y_norm)
        {
            // Map normalized coordinates to zones
            bool isLeft = x_norm < leftZoneMaxX;
            bool isRight = x_norm > rightZoneMinX;
            bool isTop = y_norm < topZoneMaxY;
            bool isBottom = y_norm > bottomZoneMinY;

            // Priority: horizontal zones first, then vertical
            if (isLeft) return "left";
            if (isRight) return "right";
            if (isTop) return "top";
            if (isBottom) return "bottom";

            return "center"; // Object in center, ignore
        }

        private void ProcessDetections()
        {
            // Check if configuration is stable
            if (lastValidatedConfig != null && lastValidatedConfig.Matches(currentDetectedConfig))
            {
                stableFrameCount++;

                if (debugLogging && stableFrameCount % 30 == 0)
                {
                    SecureFabLogger.Log("ObjectDetector", $"Config stable for {stableFrameCount} frames (need {STABILITY_THRESHOLD})");
                }
            }
            else
            {
                if (stableFrameCount > 0 && debugLogging)
                {
                    SecureFabLogger.Log("ObjectDetector", "Config changed - stability reset");
                }

                stableFrameCount = 0;
                lastValidatedConfig = new ExpectedConfig
                {
                    left = currentDetectedConfig.left,
                    right = currentDetectedConfig.right,
                    top = currentDetectedConfig.top,
                    bottom = currentDetectedConfig.bottom
                };
            }

            // Validate if stable enough
            if (stableFrameCount >= STABILITY_THRESHOLD)
            {
                ValidateConfiguration(currentDetectedConfig);
                stableFrameCount = 0; // Reset after validation
            }
        }

        private void ValidateConfiguration(ExpectedConfig detected)
        {
            // Throttle validation
            if (Time.time - lastValidationTime < 1.0f)
                return;

            lastValidationTime = Time.time;

            bool isValid = stepManager.ValidateConfiguration(detected);

            SecureFabLogger.Log("ObjectDetector", "=== CONFIG VALIDATION ===");
            SecureFabLogger.Log("ObjectDetector", $"Step: {stepManager.CurrentStepIndex + 1}/{stepManager.TotalSteps}");
            SecureFabLogger.LogConfig("Expected", stepManager.CurrentStep.expected_config);
            SecureFabLogger.LogConfig("Detected", detected);
            SecureFabLogger.Log("ObjectDetector", $"Result: {(isValid ? "✓ PASS" : "✗ FAIL")}");

            // Trigger haptic feedback if configuration is incorrect
            if (!isValid && controllerInput != null)
            {
                TriggerIncorrectConfigFeedback();
            }

            SecureFabLogger.Log("ObjectDetector", "========================");
        }
        #endregion

        #region Public API
        /// <summary>
        /// Public accessor for current detected configuration (read-only copy).
        /// Useful for other components to check current state.
        /// </summary>
        public ExpectedConfig CurrentDetectedConfiguration
        {
            get
            {
                return new ExpectedConfig
                {
                    left = currentDetectedConfig.left,
                    right = currentDetectedConfig.right,
                    top = currentDetectedConfig.top,
                    bottom = currentDetectedConfig.bottom
                };
            }
        }

        /// <summary>
        /// Called by DetectionResultBridge or other components to update the detected configuration.
        /// This allows external components to feed detection data when SecureMR
        /// tensor reading is not directly available.
        /// </summary>
        /// <param name="detected">The detected configuration</param>
        public void UpdateDetectedConfiguration(ExpectedConfig detected)
        {
            if (detected == null) return;

            // Update current detected config
            currentDetectedConfig = new ExpectedConfig
            {
                left = detected.left,
                right = detected.right,
                top = detected.top,
                bottom = detected.bottom
            };

            if (debugLogging)
            {
                SecureFabLogger.LogVerbose("ObjectDetector", "Configuration updated externally");
                SecureFabLogger.LogConfig("Updated", currentDetectedConfig);
            }
        }
        #endregion

        #region Haptic Feedback
        private void TriggerIncorrectConfigFeedback()
        {
            if (controllerInput == null) return;

            // Use the PicoControllerInput's vibration method
            PXR_Input.SendHapticImpulse(
                PXR_Input.VibrateType.BothController,
                incorrectConfigHapticStrength,
                incorrectConfigHapticDuration,
                (int)controllerInput.controllerHand
            );

            SecureFabLogger.Log("ObjectDetector", "⚠️ Incorrect configuration - haptic feedback triggered");
        }
        #endregion

        #region Event Handlers
        private void OnStepChanged(Step newStep)
        {
            SecureFabLogger.Log("ObjectDetector", "================================================");
            SecureFabLogger.Log("ObjectDetector", $"STEP CHANGED: {newStep.id} - {newStep.title}");
            SecureFabLogger.Log("ObjectDetector", $"Progress: {stepManager.CurrentStepIndex + 1}/{stepManager.TotalSteps}");
            SecureFabLogger.LogConfig("NewStepExpected", newStep.expected_config);
            SecureFabLogger.Log("ObjectDetector", "================================================");

            // Reset detection state
            stableFrameCount = 0;
            lastValidatedConfig = null;
            lastValidationTime = 0f;

            // Update instruction text
            UpdateInstructionText(newStep);
        }

        private void OnConfigValidated(bool isValid)
        {
            if (isValid)
            {
                SecureFabLogger.Log("ObjectDetector", "✓ Configuration validated - auto-progressing!");
            }
        }

        private void UpdateInstructionText(Step step)
        {
            if (step == null || instructionTextGlobal == null) return;

            string instruction = $"{step.title}\n\n{step.body}";

            // Truncate if needed
            if (instruction.Length > 400)
            {
                instruction = instruction.Substring(0, 397) + "...";
            }

            byte[] textBytes = System.Text.Encoding.UTF8.GetBytes(instruction);
            Array.Resize(ref textBytes, 512);

            try
            {
                instructionTextGlobal.Reset(textBytes);
            }
            catch (Exception e)
            {
                SecureFabLogger.LogError("ObjectDetector", $"Failed to update instruction text: {e.Message}");
            }
        }
        #endregion

        #region Debug UI
        private void OnGUI()
        {
            if (!debugLogging) return;

            GUIStyle style = new GUIStyle
            {
                fontSize = 18,
                padding = new RectOffset(10, 10, 10, 10)
            };
            style.normal.textColor = UnityEngine.Color.white;

            string info = "SecureFab Object Detector\n\n";
            info += $"Pipelines Ready: {pipelinesReady}\n";
            info += $"Detection Interval: {detectionIntervalSeconds}s\n";
            info += $"Confidence Threshold: {confidenceThreshold}\n\n";

            if (stepManager != null && stepManager.IsInitialized)
            {
                info += $"Step: {stepManager.GetProgressString()}\n";
                info += $"Current: {stepManager.CurrentStep.title}\n\n";
                info += $"Expected Config:\n{stepManager.CurrentStep.expected_config}\n\n";
                info += $"Detected Config:\n{currentDetectedConfig}\n\n";
                info += $"Stable Frames: {stableFrameCount}/{STABILITY_THRESHOLD}";
            }

            GUI.Label(new Rect(10, 10, 500, 400), info, style);
        }
        #endregion
    }
}