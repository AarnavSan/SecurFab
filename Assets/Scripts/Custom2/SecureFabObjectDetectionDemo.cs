using System;
using System.Collections.Generic;
using System.Threading;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;

namespace SecureFab.Training
{
    /// <summary>
    /// SecureMR-based object detection demo for SecureFab training.
    /// Detects objects (bottle, cup, scissors, book) and displays labels.
    /// Based on SecureMR UFODemo and YOLODemo patterns.
    /// </summary>
    public class SecureFabObjectDetectionDemo : MonoBehaviour
    {
        #region Inspector Configuration
        [Header("Models & Assets")]
        [Tooltip("YOLO model (QNN binary format)")]
        public TextAsset yoloModel;

        [Tooltip("GLTF model for instruction panel")]
        public TextAsset instructionPanelGltf;

        [Header("Pipeline Settings")]
        [Tooltip("VST image resolution")]
        public int vstWidth = 640;
        public int vstHeight = 640;

        [Tooltip("Frames to run (-1 for infinite)")]
        public int numFramesToRun = -1;

        [Tooltip("Pipeline execution interval (seconds)")]
        [Range(0.01f, 1f)]
        public float intervalBetweenPipelineRuns = 0.033f; // ~30 FPS

        [Header("Detection Settings")]
        [Tooltip("YOLO confidence threshold")]
        [Range(0f, 1f)]
        public float confidenceThreshold = 0.5f;

        [Tooltip("Maximum number of detections to process")]
        public int maxDetections = 10;

        [Header("Training Integration")]
        [Tooltip("Reference to StepManager")]
        public StepManager stepManager;

        [Header("Debug")]
        public bool debugLogging = true;
        #endregion

        #region COCO Class ID Mapping
        // COCO dataset class IDs for our training objects
        private const int CLASS_BOTTLE = 39;
        private const int CLASS_CUP = 41;
        private const int CLASS_SCISSORS = 76;
        private const int CLASS_BOOK = 73;

        private readonly Dictionary<int, string> classIdToName = new Dictionary<int, string>
        {
            { CLASS_BOTTLE, "bottle" },
            { CLASS_CUP, "cup" },
            { CLASS_SCISSORS, "scissors" },
            { CLASS_BOOK, "book" }
        };
        #endregion

        #region SecureMR Pipeline Components
        private Provider provider;
        private Pipeline vstPipeline;
        private Pipeline detectionPipeline;
        private Pipeline renderPipeline;

        // VST (Video See Through) tensors
        private Tensor vstImageTensor;

        // Detection tensors
        private Tensor detectionOutputTensor;

        // Render tensors
        private Tensor instructionTextTensor;
        private Tensor panelGltfTensor;
        private Tensor panelTransformTensor;

        // Placeholders for tensor references
        private Tensor vstImagePlaceholder;
        private Tensor detectionOutputPlaceholder;
        private Tensor instructionTextPlaceholder;
        private Tensor panelGltfPlaceholder;

        // Threading
        private Thread pipelineThread;
        private bool isRunning = false;
        private int frameCount = 0;
        private float elapsedTime = 0f;
        #endregion

        #region Unity Lifecycle
        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        private void Start()
        {
            InitializeSecureMR();
        }

        private void OnDestroy()
        {
            Cleanup();
        }

        private void Update()
        {
            if (!isRunning) return;

            // Update instruction display
            if (stepManager != null && stepManager.CurrentStep != null)
            {
                UpdateInstructionDisplay(stepManager.CurrentStep);
            }

            // Run pipelines at interval
            elapsedTime += Time.deltaTime;
            if (elapsedTime < intervalBetweenPipelineRuns) return;
            elapsedTime = 0f;

            // Check frame limit
            if (numFramesToRun >= 0 && frameCount >= numFramesToRun)
            {
                isRunning = false;
                return;
            }

            try
            {
                // Run VST pipeline (camera capture)
                vstPipeline.Execute(new TensorMapping());

                // Run detection pipeline (YOLO inference)
                var detectionMapping = new TensorMapping();
                detectionMapping.Set(vstImagePlaceholder, vstImageTensor);
                detectionPipeline.Execute(detectionMapping);

                // Run render pipeline (display instructions)
                var renderMapping = new TensorMapping();
                renderMapping.Set(instructionTextPlaceholder, instructionTextTensor);
                if (panelGltfPlaceholder != null && panelGltfTensor != null)
                {
                    renderMapping.Set(panelGltfPlaceholder, panelGltfTensor);
                }
                renderPipeline.Execute(renderMapping);

                frameCount++;
            }
            catch (Exception e)
            {
                LogError($"Pipeline error: {e.Message}");
            }
        }
        #endregion

        #region Initialization
        private void InitializeSecureMR()
        {
            try
            {
                LogDebug("Initializing SecureMR pipelines...");

                // Enable video see-through
                PXR_Manager.EnableVideoSeeThrough = true;

                // Create provider
                provider = new Provider(vstWidth, vstHeight);
                LogDebug("Provider created");

                // Create pipelines
                CreateVSTPipeline();
                CreateDetectionPipeline();
                CreateRenderPipeline();

                isRunning = true;

                LogDebug("✓ SecureMR initialization complete");
            }
            catch (Exception e)
            {
                LogError($"Failed to initialize SecureMR: {e.Message}\n{e.StackTrace}");
            }
        }

        private void CreateVSTPipeline()
        {
            LogDebug("Creating VST pipeline...");

            vstPipeline = provider.CreatePipeline();

            // Create VST operator for camera passthrough
            var vstOp = vstPipeline.CreateOperator<RectifiedVstAccessOperator>();
            
            // Create output tensor for VST image (3 channels, height x width)
            vstImageTensor = provider.CreateTensor<byte, Matrix>(
                3,
                new TensorShape(new[] { vstHeight, vstWidth })
            );

            vstOp.SetResult("left image", vstImageTensor);

            LogDebug("✓ VST pipeline created");
        }

        private void CreateDetectionPipeline()
        {
            LogDebug("Creating detection pipeline...");

            if (yoloModel == null)
            {
                LogError("YOLO model not assigned!");
                return;
            }

            detectionPipeline = provider.CreatePipeline();
            
            // Input: VST image placeholder (will be mapped from vstImageTensor)
            vstImagePlaceholder = detectionPipeline.CreateTensorReference<byte, Matrix>(
                3,
                new TensorShape(new[] { vstHeight, vstWidth })
            );
            
            // Convert uint8 to float32
            var vstImageFloat = detectionPipeline.CreateTensor<float, Matrix>(
                3,
                new TensorShape(new[] { vstHeight, vstWidth })
            );
            
            var assignOp = detectionPipeline.CreateOperator<AssignmentOperator>();
            assignOp.SetOperand("src", vstImagePlaceholder);
            assignOp.SetResult("dst", vstImageFloat);
            
            // Normalize to [0, 1]
            var normalizeOp = detectionPipeline.CreateOperator<ArithmeticComposeOperator>(
                new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));
            normalizeOp.SetOperand("{0}", vstImageFloat);
            normalizeOp.SetResult("result", vstImageFloat);

            // YOLO model configuration
            var modelConfig = new ModelOperatorConfiguration(
                yoloModel.bytes,
                SecureMRModelType.QnnContextBinary,
                "yolo"
            );

            // Configure input/output mappings
            // NOTE: These names must match your YOLO model's tensor names
            modelConfig.AddInputMapping("images", "images", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("output0", "output0", SecureMRModelEncoding.Float32);

            var modelOp = detectionPipeline.CreateOperator<RunModelInferenceOperator>(modelConfig);
            modelOp.SetOperand("images", vstImageFloat);

            // Raw YOLO output - YOLOv8 format [84, 8400]
            var rawDetections = detectionPipeline.CreateTensor<float, Matrix>(
                1,
                new TensorShape(new[] { 84, 8400 })
            );
            modelOp.SetResult("output0", rawDetections);

            // Processed detections output
            // Format: [x_center, y_center, width, height, confidence, class_id]
            detectionOutputTensor = provider.CreateTensor<float, Matrix>(
                1,
                new TensorShape(new[] { maxDetections, 6 })
            );

            // Post-process YOLO output (confidence filtering, NMS)
            // NOTE: In a complete implementation, you'd add NMS operators here
            // For now, we'll do post-processing in Unity
            ProcessYOLOOutput(rawDetections, detectionOutputTensor);

            LogDebug("✓ Detection pipeline created");
        }

        private void ProcessYOLOOutput(Tensor rawOutput, Tensor processedOutput)
        {
            // NOTE: This is simplified for the hackathon
            // In a production system, you'd implement:
            // 1. Confidence thresholding using comparison operators
            // 2. Class filtering (only keep relevant classes)
            // 3. NMS (Non-Maximum Suppression) using existing SecureMR operators
            // 4. Format conversion to [x, y, w, h, conf, class]

            // For this demo, we'll process detections in Unity after reading from global tensor
            LogDebug("YOLO post-processing placeholder (simplified for hackathon)");
        }

        private void CreateRenderPipeline()
        {
            LogDebug("Creating render pipeline...");

            renderPipeline = provider.CreatePipeline();

            // Instruction text tensor (512 bytes for UTF-8 text)
            instructionTextTensor = provider.CreateTensor<byte, Scalar>(
                1,
                new TensorShape(new[] { 512 })
            );

            // Text placeholder for pipeline
            instructionTextPlaceholder = renderPipeline.CreateTensorReference<byte, Scalar>(
                1,
                new TensorShape(new[] { 512 })
            );

            // Text rendering operator (like SecureMRSample)
            var textConfig = new RenderTextOperatorConfiguration(
                SecureMRFontTypeface.SansSerif,
                "en-US",
                1440,
                960
            );
            var textOp = renderPipeline.CreateOperator<RenderTextOperator>(textConfig);
            textOp.SetOperand("text", instructionTextPlaceholder);

            // Text position
            var startPosition = renderPipeline.CreateTensor<float, Point>(
                2,
                new TensorShape(new[] { 1 }),
                new float[] { 0.1f, 0.3f }
            );
            textOp.SetOperand("start", startPosition);

            // Text colors (foreground, background)
            var colors = renderPipeline.CreateTensor<byte, Unity.XR.PXR.SecureMR.Color>(
                4,
                new TensorShape(new[] { 2 }),
                new byte[] {
                    255, 255, 255, 255,  // White text
                    0, 0, 0, 255         // Black background
                }
            );
            textOp.SetOperand("colors", colors);

            var textureId = renderPipeline.CreateTensor<ushort, Scalar>(
                1,
                new TensorShape(new[] { 1 }),
                new ushort[] { 0 }
            );
            textOp.SetOperand("texture ID", textureId);

            var fontSize = renderPipeline.CreateTensor<float, Scalar>(
                1,
                new TensorShape(new[] { 1 }),
                new float[] { 72.0f }
            );
            textOp.SetOperand("font size", fontSize);

            // GLTF panel for background
            if (instructionPanelGltf != null)
            {
                panelGltfTensor = provider.CreateTensor<Gltf>(instructionPanelGltf.bytes);
                panelGltfPlaceholder = renderPipeline.CreateTensorReference<Gltf>();

                // Panel transform (position in world space)
                panelTransformTensor = renderPipeline.CreateTensor<float, Matrix>(
                    1,
                    new TensorShape(new[] { 4, 4 }),
                    new float[]
                    {
                        0.5f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.5f, 0.0f, 0.25f,
                        0.0f, 0.0f, 0.5f, -1.5f,
                        0.0f, 0.0f, 0.0f, 1.0f
                    }
                );

                // Render GLTF operator
                var renderGltfOp = renderPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
                renderGltfOp.SetOperand("gltf", panelGltfPlaceholder);
                renderGltfOp.SetOperand("world pose", panelTransformTensor);
                
                // Connect text to GLTF
                textOp.SetOperand("gltf", panelGltfPlaceholder);
            }

            LogDebug("✓ Render pipeline created");
        }
        #endregion

        #region Pipeline Execution
        private void ProcessDetections()
        {
            try
            {
                // Read detections from global tensor
                // NOTE: In SecureMR, we can't directly read tensors in Unity
                // This is a simplified version for demonstration

                // In a real implementation, you would:
                // 1. Create a separate pipeline that writes detection results
                //    to a format Unity can read (e.g., via RenderTextOperator)
                // 2. Use indirect signals (GLTF visibility, text labels) to
                //    communicate detection results
                // 3. Keep all tensor reading inside SecureMR pipelines

                LogDebug($"Frame {frameCount}: Processing detections");

                // For this hackathon demo, we'll use the detection output
                // to update labels through the render pipeline
            }
            catch (Exception e)
            {
                LogError($"Failed to process detections: {e.Message}");
            }
        }
        #endregion

        #region Label Display
        private void UpdateInstructionDisplay(Step step)
        {
            if (step == null || instructionTextTensor == null) return;

            // Format instruction text
            string instruction = FormatInstruction(step);

            // Convert to UTF-8 bytes
            byte[] textBytes = System.Text.Encoding.UTF8.GetBytes(instruction);
            Array.Resize(ref textBytes, 512); // Pad to tensor size

            try
            {
                instructionTextTensor.Reset(textBytes);
            }
            catch (Exception e)
            {
                LogError($"Failed to update instruction text: {e.Message}");
            }
        }

        private string FormatInstruction(Step step)
        {
            string text = $"{step.title}\n\n{step.body}";

            // Add expected configuration
            if (step.expected_config != null)
            {
                text += "\n\nExpected Configuration:";
                if (!string.IsNullOrEmpty(step.expected_config.left))
                    text += $"\nLEFT: {step.expected_config.left}";
                if (!string.IsNullOrEmpty(step.expected_config.right))
                    text += $"\nRIGHT: {step.expected_config.right}";
                if (!string.IsNullOrEmpty(step.expected_config.top))
                    text += $"\nTOP: {step.expected_config.top}";
                if (!string.IsNullOrEmpty(step.expected_config.bottom))
                    text += $"\nBOTTOM: {step.expected_config.bottom}";
            }

            // Truncate if too long
            if (text.Length > 400)
            {
                text = text.Substring(0, 397) + "...";
            }

            return text;
        }

        /// <summary>
        /// Update detection labels based on YOLO output.
        /// NOTE: In a complete implementation, this would be done entirely
        /// within SecureMR using RenderTextOperator for each detected object.
        /// </summary>
        private void UpdateDetectionLabels(List<Detection> detections)
        {
            if (detections == null || detections.Count == 0) return;

            // For each detection, you would create a text label
            // using RenderTextOperator positioned at the detection location
            // This is simplified for the hackathon - full implementation
            // would require dynamic text rendering in SecureMR

            LogDebug($"Detected {detections.Count} objects:");
            foreach (var det in detections)
            {
                string className = classIdToName.ContainsKey(det.classId) 
                    ? classIdToName[det.classId] 
                    : $"class_{det.classId}";
                
                LogDebug($"  - {className} at ({det.x:F2}, {det.y:F2}) " +
                        $"conf={det.confidence:F2}");
            }
        }
        #endregion

        #region Cleanup
        private void Cleanup()
        {
            LogDebug("Cleaning up SecureMR resources...");
            isRunning = false;
            LogDebug("✓ Cleanup complete");
        }
        #endregion

        #region Utilities
        private void LogDebug(string message)
        {
            if (debugLogging)
            {
                Debug.Log($"[SecureFabDemo] {message}");
            }
        }

        private void LogError(string message)
        {
            Debug.LogError($"[SecureFabDemo] {message}");
        }
        #endregion

        #region Helper Structures
        private struct Detection
        {
            public float x;          // Center X (normalized 0-1)
            public float y;          // Center Y (normalized 0-1)
            public float width;      // Width (normalized 0-1)
            public float height;     // Height (normalized 0-1)
            public float confidence; // Detection confidence
            public int classId;      // COCO class ID

            public Detection(float x, float y, float w, float h, int cls, float conf)
            {
                this.x = x;
                this.y = y;
                this.width = w;
                this.height = h;
                this.classId = cls;
                this.confidence = conf;
            }
        }
        #endregion

        #region Manual Testing (Debug Only)
        private void OnGUI()
        {
            if (!debugLogging) return;

            GUIStyle style = new GUIStyle
            {
                fontSize = 16,
                padding = new RectOffset(10, 10, 10, 10)
            };
            style.normal.textColor = UnityEngine.Color.white;

            string info = "SecureFab Object Detection Demo\n\n";
            info += $"Frame: {frameCount}\n";
            info += $"VST: {vstWidth}x{vstHeight}\n";
            info += $"Confidence: {confidenceThreshold}\n\n";

            if (stepManager != null && stepManager.CurrentStep != null)
            {
                info += $"Current Step: {stepManager.CurrentStep.title}\n";
            }

            GUI.Label(new Rect(10, 10, 400, 300), info, style);
        }
        #endregion
    }
}