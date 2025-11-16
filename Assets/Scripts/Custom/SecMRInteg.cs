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
    /// SIMPLIFIED VERSION - Uses only basic SecureMR operators
    /// This version focuses on getting a working demo without advanced detection features
    /// </summary>
    public class SecMRInteg : MonoBehaviour
    {
        #region Original Fields
        public TextAsset helmetGltfAsset;
        public int vstWidth = 1024;
        public int vstHeight = 1024;

        private Provider provider;
        private Pipeline pipeline;
        private Tensor gltfTensor;
        private Tensor gltfPlaceholderTensor;
        #endregion

        #region New Training Demo Fields
        [Header("Training Configuration")]
        [Tooltip("YOLO model for object detection")]
        public TextAsset yoloModel;

        [Tooltip("GLTF asset for instruction panel background")]
        public TextAsset instructionPanelGltf;

        [Tooltip("Reference to StepManager")]
        public StepManager stepManager;

        [Header("Zone Thresholds (Normalized 0-1)")]
        [Range(0f, 0.5f)]
        public float leftZoneX = 0.33f;

        [Range(0.5f, 1f)]
        public float rightZoneX = 0.66f;

        [Range(0f, 0.5f)]
        public float topZoneY = 0.33f;

        [Range(0.5f, 1f)]
        public float bottomZoneY = 0.66f;

        [Header("Detection Settings")]
        [Range(0f, 1f)]
        public float confidenceThreshold = 0.5f;

        public int maxDetections = 4;

        [Header("Manual Testing")]
        [Tooltip("Enable keyboard controls for testing without objects")]
        public bool enableManualControls = true;

        [Header("Debug")]
        public bool debugLogging = true;

        // COCO class IDs for training objects
        private const int COCO_BOTTLE = 39;
        private const int COCO_CUP = 41;
        private const int COCO_SCISSORS = 76;
        private const int COCO_BOOK = 73;

        // Object mapping
        private readonly Dictionary<int, string> trainingObjectMap = new Dictionary<int, string>
        {
            { COCO_BOTTLE, "bottle" },
            { COCO_CUP, "cup" },
            { COCO_SCISSORS, "scissors" },
            { COCO_BOOK, "book" }
        };

        // Pipelines
        private Pipeline vstPipeline;
        private Pipeline inferencePipeline;
        private Pipeline textRenderPipeline;

        // Global tensors
        private Tensor vstOutputLeftFp32Global;
        private Tensor vstTimestampGlobal;
        private Tensor instructionTextGlobal;
        private Tensor instructionPanelGltfTensor;

        // Detection state
        private ExpectedConfig currentDetectedConfig = new ExpectedConfig();
        private ExpectedConfig lastStableConfig;
        private int stableFrameCount = 0;
        private float lastValidationTime = 0f;
        private int stabilityFrames = 10;

        // Threading
        private Thread vstThread;
        private bool keepRunning = true;
        private bool pipelinesReady = false;
        private readonly object initLock = new object();

        #endregion

        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        // In your ACTUAL SecMRInteg.cs Start() method
        private void Start()
        {
            // Test log to verify logcat visibility
            SecureFabLogger.Log("STARTUP", "╔════════════════════════════════════╗");
            SecureFabLogger.Log("STARTUP", "║  SecureFab Training Demo Starting  ║");
            SecureFabLogger.Log("STARTUP", "╚════════════════════════════════════╝");
            
            CreateProvider();
            CreateGlobals();
            CreatePipeline();
            
            // Initialize training components
            InitializeTrainingComponents();
            
            SecureFabLogger.Log("STARTUP", "Start() method complete");
        }

        private void Update()
        {
            RunPipeline();

            // Handle manual keyboard controls for testing
            if (enableManualControls)
            {
                HandleManualControls();
            }

            // Process detections and validate configuration
            if (pipelinesReady && stepManager != null)
            {
                ProcessDetections();
            }
        }

        private void OnDestroy()
        {
            keepRunning = false;
            vstThread?.Join(1000);
        }

        private void CreateProvider()
        {
            provider = new Provider(vstWidth, vstHeight);
        }

        private void CreateGlobals()
        {
            // Create GLTF tensor
            gltfTensor = provider.CreateTensor<Gltf>(helmetGltfAsset.bytes);

            // Create training-specific global tensors (if models assigned)
            if (yoloModel != null && instructionPanelGltf != null)
            {
                vstOutputLeftFp32Global = provider.CreateTensor<float, Matrix>(
                    3, new TensorShape(new[] { vstHeight, vstWidth }));
                vstTimestampGlobal = provider.CreateTensor<int, TimeStamp>(
                    4, new TensorShape(new[] { 1 }));

                instructionTextGlobal = provider.CreateTensor<byte, Scalar>(
                    1, new TensorShape(new[] { 512 }));  // Larger buffer for text
                instructionPanelGltfTensor = provider.CreateTensor<Gltf>(instructionPanelGltf.bytes);
            }
        }

        private void CreatePipeline()
        {
            pipeline = provider.CreatePipeline();

            // Create transform matrix tensor
            int[] transformDim = { 4, 4 };
            var transformShape = new TensorShape(transformDim);
            float[] transformData = {
                0.5f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.5f, 0.0f, 0.25f,
                0.0f, 0.0f, 0.5f, -1.5f,
                0.0f, 0.0f, 0.0f, 1.0f
            };
            var poseTensor = pipeline.CreateTensor<float, Matrix>(1, transformShape, transformData);

            // Create GLTF tensor placeholder
            gltfPlaceholderTensor = pipeline.CreateTensorReference<Gltf>();

            // Create render GLTF operator
            var renderGltfOperator = pipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            renderGltfOperator.SetOperand("gltf", gltfPlaceholderTensor);
            renderGltfOperator.SetOperand("world pose", poseTensor);
        }

        private void RunPipeline()
        {
            if (debugLogging && Time.frameCount % 300 == 0)  // Log every 300 frames
            {
                Debug.Log("[SecMRInteg] Running pipeline...");
            }

            var tensorMapping = new TensorMapping();
            tensorMapping.Set(gltfPlaceholderTensor, gltfTensor);
            pipeline.Execute(tensorMapping);
        }

        #region Training Components Initialization

        // In your ACTUAL SecMRInteg.cs
        private void InitializeTrainingComponents()
        {
            SecureFabLogger.Log("SecMRInteg", "=== INITIALIZATION START ===");

            if (yoloModel == null || instructionPanelGltf == null)
            {
                SecureFabLogger.LogWarning("SecMRInteg", "YOLO model or instruction panel GLTF not assigned. Training features disabled.");
                SecureFabLogger.Log("SecMRInteg", "Manual keyboard controls enabled. Use keys 1-5 to test steps.");
                return;
            }

            if (stepManager == null)
            {
                SecureFabLogger.LogError("SecMRInteg", "StepManager reference missing!");
                return;
            }

            if (!stepManager.IsInitialized)
            {
                SecureFabLogger.LogError("SecMRInteg", "StepManager not initialized!");
                return;
            }

            // Subscribe to step manager events
            stepManager.onStepChanged.AddListener(OnStepChanged);
            stepManager.onConfigurationValidated.AddListener(OnConfigValidated);

            SecureFabLogger.Log("SecMRInteg", "Starting pipeline creation in background...");

            // Start pipeline creation in background
            Thread pipelineInitThread = new Thread(() =>
            {
                try
                {
                    SecureFabLogger.Log("SecMRInteg", "Creating simplified pipelines...");
                    CreateSimplifiedPipelines();

                    lock (initLock)
                    {
                        pipelinesReady = true;
                    }

                    SecureFabLogger.Log("SecMRInteg", "✓ Training pipelines initialized successfully!");
                }
                catch (Exception e)
                {
                    SecureFabLogger.LogError("SecMRInteg", $"Pipeline init FAILED: {e.Message}\n{e.StackTrace}");
                }
            });

            pipelineInitThread.Start();

            SecureFabLogger.Log("SecMRInteg", $"Current step: {stepManager.CurrentStep.title}");
            SecureFabLogger.Log("SecMRInteg", "=== INITIALIZATION COMPLETE ===");

            // Update instruction text for first step
            UpdateInstructionText(stepManager.CurrentStep);
        }
        #endregion

        #region Simplified Pipelines

        private void CreateSimplifiedPipelines()
        {
            LogDebug("Creating simplified VST pipeline...");
            CreateSimplifiedVSTPipeline();

            LogDebug("Creating text render pipeline...");
            CreateTextRenderPipeline();

            LogDebug("Simplified pipelines created!");
        }

        private void CreateSimplifiedVSTPipeline()
        {
            vstPipeline = provider.CreatePipeline();

            var vstOutputLeftFp32Placeholder = vstPipeline.CreateTensorReference<float, Matrix>(
                3, new TensorShape(new[] { vstHeight, vstWidth }));
            var vstTimestampPlaceholder = vstPipeline.CreateTensorReference<int, TimeStamp>(
                4, new TensorShape(new[] { 1 }));

            var vstOutputLeftUint8 = vstPipeline.CreateTensor<byte, Matrix>(
                3, new TensorShape(new[] { vstHeight, vstWidth }));
            var vstOutputRightUint8 = vstPipeline.CreateTensor<byte, Matrix>(
                3, new TensorShape(new[] { vstHeight, vstWidth }));
            var vstCameraMatrix = vstPipeline.CreateTensor<float, Matrix>(
                1, new TensorShape(new[] { 3, 3 }));

            var vstOp = vstPipeline.CreateOperator<RectifiedVstAccessOperator>();
            vstOp.SetResult("left image", vstOutputLeftUint8);
            vstOp.SetResult("right image", vstOutputRightUint8);
            vstOp.SetResult("timestamp", vstTimestampPlaceholder);
            vstOp.SetResult("camera matrix", vstCameraMatrix);

            var assignOp = vstPipeline.CreateOperator<AssignmentOperator>();
            assignOp.SetOperand("src", vstOutputLeftUint8);
            assignOp.SetResult("dst", vstOutputLeftFp32Placeholder);

            var normalizeOp = vstPipeline.CreateOperator<ArithmeticComposeOperator>(
                new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));
            normalizeOp.SetOperand("{0}", vstOutputLeftFp32Placeholder);
            normalizeOp.SetResult("result", vstOutputLeftFp32Placeholder);
        }

        #endregion

        #region Text Rendering Pipeline

        private void CreateTextRenderPipeline()
        {
            textRenderPipeline = provider.CreatePipeline();

            var instructionTextPlaceholder = textRenderPipeline.CreateTensorReference<byte, Scalar>(
                1, new TensorShape(new[] { 512 }));
            var panelGltfPlaceholder = textRenderPipeline.CreateTensorReference<Gltf>();

            // Position panel in front of user (closer and larger for visibility)
            int[] transformDim = { 4, 4 };
            var transformShape = new TensorShape(transformDim);
            float[] panelTransformData = {
                0.5f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.5f, 0.0f, 0.3f,
                0.0f, 0.0f, 0.5f, -1.2f,
                0.0f, 0.0f, 0.0f, 1.0f
            };
            var panelPoseTensor = textRenderPipeline.CreateTensor<float, Matrix>(1, transformShape, panelTransformData);

            // Render text on panel
            var startPosition = textRenderPipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }));
            startPosition.Reset(new float[] { 0.1f, 0.5f });

            var colors = textRenderPipeline.CreateTensor<byte, Unity.XR.PXR.SecureMR.Color>(
                4, new TensorShape(new[] { 2 }));
            colors.Reset(new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 });

            var textureId = textRenderPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }));
            textureId.Reset(new ushort[] { 0 });

            var fontSize = textRenderPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
            fontSize.Reset(new float[] { 64.0f });

            var renderTextOp = textRenderPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1024, 1024));
            renderTextOp.SetOperand("text", instructionTextPlaceholder);
            renderTextOp.SetOperand("start", startPosition);
            renderTextOp.SetOperand("colors", colors);
            renderTextOp.SetOperand("texture ID", textureId);
            renderTextOp.SetOperand("font size", fontSize);
            renderTextOp.SetOperand("gltf", panelGltfPlaceholder);

            var renderPanelOp = textRenderPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            renderPanelOp.SetOperand("gltf", panelGltfPlaceholder);
            renderPanelOp.SetOperand("world pose", panelPoseTensor);
        }

        private void RunTextRenderPipeline()
        {
            if (textRenderPipeline == null) return;

            try
            {
                var tensorMapping = new TensorMapping();
                tensorMapping.Set(textRenderPipeline.CreateTensorReference<byte, Scalar>(1, new TensorShape(new[] { 512 })),
                                 instructionTextGlobal);
                tensorMapping.Set(textRenderPipeline.CreateTensorReference<Gltf>(),
                                 instructionPanelGltfTensor);

                textRenderPipeline.Execute(tensorMapping);
            }
            catch (Exception e)
            {
                Debug.LogError($"[SecMRInteg] Text render error: {e.Message}");
            }
        }

        #endregion

        #region Manual Controls (for testing without objects)

        // In your ACTUAL SecMRInteg.cs
        private void HandleManualControls()
        {
            if (stepManager == null || !stepManager.IsInitialized) return;

            // Number keys 1-5 to jump to specific steps
            if (Input.GetKeyDown(KeyCode.Alpha1))
            {
                SecureFabLogger.Log("SecMRInteg", "Manual: Jump to Step 1");
                TestConfigForStep(0);
            }
            if (Input.GetKeyDown(KeyCode.Alpha2))
            {
                SecureFabLogger.Log("SecMRInteg", "Manual: Jump to Step 2");
                TestConfigForStep(1);
            }
            if (Input.GetKeyDown(KeyCode.Alpha3))
            {
                SecureFabLogger.Log("SecMRInteg", "Manual: Jump to Step 3");
                TestConfigForStep(2);
            }
            if (Input.GetKeyDown(KeyCode.Alpha4))
            {
                SecureFabLogger.Log("SecMRInteg", "Manual: Jump to Step 4");
                TestConfigForStep(3);
            }
            if (Input.GetKeyDown(KeyCode.Alpha5))
            {
                SecureFabLogger.Log("SecMRInteg", "Manual: Jump to Step 5");
                TestConfigForStep(4);
            }

            // Arrow keys for navigation
            if (Input.GetKeyDown(KeyCode.RightArrow))
            {
                SecureFabLogger.Log("SecMRInteg", "Manual: Next Step");
                stepManager.GoToNextStep();
            }
            if (Input.GetKeyDown(KeyCode.LeftArrow))
            {
                SecureFabLogger.Log("SecMRInteg", "Manual: Previous Step");
                stepManager.GoToPreviousStep();
            }

            // Space to simulate correct config
            if (Input.GetKeyDown(KeyCode.Space))
            {
                SecureFabLogger.Log("SecMRInteg", "Manual: Simulating correct configuration");
                SimulateCorrectConfiguration();
            }

            // R to reset to first step
            if (Input.GetKeyDown(KeyCode.R))
            {
                SecureFabLogger.Log("SecMRInteg", "Manual: Reset to first step");
                stepManager.ResetToFirstStep();
            }
        }

        private void TestConfigForStep(int stepId)
        {
            var step = stepManager.GetStepById(stepId);
            if (step != null)
            {
                stepManager.SetStepById(stepId);
                LogDebug($"Jumped to Step {stepId + 1}");
            }
        }

        // In your ACTUAL SecMRInteg.cs
        private void SimulateCorrectConfiguration()
        {
            if (stepManager.CurrentStep != null)
            {
                var correctConfig = stepManager.CurrentStep.expected_config;
                currentDetectedConfig = new ExpectedConfig
                {
                    left = correctConfig.left,
                    right = correctConfig.right,
                    top = correctConfig.top,
                    bottom = correctConfig.bottom
                };

                stableFrameCount = stabilityFrames; // Force validation

                // ENHANCED LOGGING
                SecureFabLogger.Log("SecMRInteg", "Simulating correct config:");
                SecureFabLogger.LogConfig("Simulated", currentDetectedConfig);
            }
        }

        #endregion

        #region Detection Processing (Simplified)

        // In your ACTUAL SecMRInteg.cs
        private void ProcessDetections()
        {
            // For now, use manual simulation until YOLO pipeline is working
            // This allows testing the step progression system

            // Check stability
            if (lastStableConfig != null && lastStableConfig.Matches(currentDetectedConfig))
            {
                stableFrameCount++;

                // Log stability progress every 30 frames
                if (debugLogging && stableFrameCount % 30 == 0)
                {
                    SecureFabLogger.Log("SecMRInteg", $"Config stable for {stableFrameCount} frames (need {stabilityFrames})");
                }
            }
            else
            {
                if (stableFrameCount > 0 && debugLogging)
                {
                    SecureFabLogger.Log("SecMRInteg", "Config changed - stability reset");
                }

                stableFrameCount = 0;
                lastStableConfig = new ExpectedConfig
                {
                    left = currentDetectedConfig.left,
                    right = currentDetectedConfig.right,
                    top = currentDetectedConfig.top,
                    bottom = currentDetectedConfig.bottom
                };
            }

            // Validate if stable
            if (stableFrameCount >= stabilityFrames)
            {
                ValidateConfiguration(currentDetectedConfig);
                stableFrameCount = 0; // Reset to avoid continuous validation
            }
        }

        // In your ACTUAL SecMRInteg.cs
        private void ValidateConfiguration(ExpectedConfig detected)
        {
            // Throttle validation to avoid spam
            if (Time.time - lastValidationTime < 1.0f)
                return;

            lastValidationTime = Time.time;

            bool isValid = stepManager.ValidateConfiguration(detected);

            // ENHANCED LOGGING
            SecureFabLogger.Log("SecMRInteg", "=== CONFIG VALIDATION ===");
            SecureFabLogger.Log("SecMRInteg", $"Step: {stepManager.CurrentStepIndex + 1}/{stepManager.TotalSteps}");
            SecureFabLogger.LogConfig("Expected", stepManager.CurrentStep.expected_config);
            SecureFabLogger.LogConfig("Detected", detected);
            SecureFabLogger.Log("SecMRInteg", $"Result: {(isValid ? "✓ PASS" : "✗ FAIL")}");
            SecureFabLogger.Log("SecMRInteg", "===================");
        }

        #endregion

        #region Event Handlers

        // In your ACTUAL SecMRInteg.cs
        private void OnStepChanged(Step newStep)
        {
            // ENHANCED LOGGING
            SecureFabLogger.Log("SecMRInteg", "================================================");
            SecureFabLogger.Log("SecMRInteg", $"STEP CHANGED: {newStep.id} - {newStep.title}");
            SecureFabLogger.Log("SecMRInteg", $"Progress: {stepManager.CurrentStepIndex + 1}/{stepManager.TotalSteps}");
            SecureFabLogger.LogConfig("NewStepExpected", newStep.expected_config);
            SecureFabLogger.Log("SecMRInteg", "================================================");

            stableFrameCount = 0;
            lastStableConfig = null;
            lastValidationTime = 0f;

            UpdateInstructionText(newStep);
        }

        private void OnConfigValidated(bool isValid)
        {
            // Optional: Add visual/audio feedback
            if (isValid)
            {
                LogDebug("✓ Step validated successfully!");
            }
        }

        private void UpdateInstructionText(Step step)
        {
            if (step == null || instructionTextGlobal == null) return;

            string instruction = $"{step.title}\n\n{step.body}";

            // Truncate if too long
            if (instruction.Length > 400)
            {
                instruction = instruction.Substring(0, 397) + "...";
            }

            byte[] textBytes = System.Text.Encoding.UTF8.GetBytes(instruction);

            // Ensure buffer size
            if (textBytes.Length > 512)
            {
                Array.Resize(ref textBytes, 512);
            }
            else
            {
                Array.Resize(ref textBytes, 512);
            }

            try
            {
                instructionTextGlobal.Reset(textBytes);

                if (pipelinesReady && textRenderPipeline != null)
                {
                    RunTextRenderPipeline();
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[SecMRInteg] Failed to update text: {e.Message}");
            }
        }

        #endregion

        #region Helper Methods

        // In SecMRInteg.cs, replace LogDebug() with:
        private void LogDebug(string message)
        {
            if (debugLogging)
            {
                SecureFabLogger.Log("SecMRInteg", message);
            }
        }

        // Add these specialized logging methods:
        private void LogDetection(string objectName, string zone, float confidence)
        {
            if (debugLogging)
            {
                SecureFabLogger.LogDetection("SecMRInteg", objectName, zone, confidence);
            }
        }

        private void LogConfig(ExpectedConfig config)
        {
            if (debugLogging)
            {
                SecureFabLogger.LogConfig("SecMRInteg", config);
            }
        }

        #endregion

        #region Debug Info Display

        private void OnGUI()
        {
            if (!debugLogging || stepManager == null || !stepManager.IsInitialized) return;

            // Display current step info in top-left corner
            GUIStyle style = new GUIStyle();
            style.fontSize = 20;
            style.normal.textColor = UnityEngine.Color.white;
            style.padding = new RectOffset(10, 10, 10, 10);

            string info = $"SecureFab Training Demo\n\n";
            info += $"Step: {stepManager.GetProgressString()}\n";
            info += $"Current: {stepManager.CurrentStep.title}\n\n";

            if (enableManualControls)
            {
                info += "MANUAL CONTROLS:\n";
                info += "  1-5: Jump to step\n";
                info += "  ←→: Prev/Next step\n";
                info += "  SPACE: Simulate correct config\n";
                info += "  R: Reset to first step\n\n";
            }

            info += $"Expected Config:\n{stepManager.CurrentStep.expected_config}\n\n";
            info += $"Detected Config:\n{currentDetectedConfig}";

            GUI.Label(new Rect(10, 10, 600, 400), info, style);
        }

        #endregion
    }
}