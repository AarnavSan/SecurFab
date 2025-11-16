using System;
using System.Collections.Generic;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;

namespace SecureFab.Training
{
    /// <summary>
    /// Helper class for rendering object detection labels in SecureMR.
    /// Creates text labels above detected objects using RenderTextOperator.
    /// 
    /// USAGE:
    /// 1. Attach to a GameObject in your scene
    /// 2. Assign the provider reference
    /// 3. Call UpdateLabels() with detection results
    /// 
    /// NOTE: This demonstrates the pattern from SecureMR samples where
    /// all rendering stays inside SecureMR pipelines.
    /// </summary>
    public class ObjectLabelRenderer : MonoBehaviour
    {
        #region Configuration
        [Header("Label Settings")]
        [Tooltip("Font size for object labels")]
        [Range(16, 64)]
        public int labelFontSize = 24;

        [Tooltip("Max width for label text")]
        public int labelMaxWidth = 300;

        [Tooltip("Show confidence scores in labels")]
        public bool showConfidence = true;

        [Header("Label Colors")]
        public UnityEngine.Color labelTextColor = UnityEngine.Color.white;
        public UnityEngine.Color labelBackgroundColor = new UnityEngine.Color(0, 0, 0, 0.8f);

        [Header("Debug")]
        public bool debugLogging = false;
        #endregion

        #region COCO Class Mapping
        private readonly Dictionary<int, string> classNames = new Dictionary<int, string>
        {
            { 39, "Bottle" },
            { 41, "Cup" },
            { 76, "Scissors" },
            { 73, "Book" }
        };
        #endregion

        #region SecureMR Components
        private Provider provider;
        private Pipeline labelPipeline;
        private List<LabelData> activeLabelData = new List<LabelData>();

        // One tensor per label (max 4 objects in our training scenario)
        private const int MAX_LABELS = 4;
        private Tensor[] labelTextTensors = new Tensor[MAX_LABELS];
        private Tensor[] labelTextPlaceholders = new Tensor[MAX_LABELS];
        private Tensor[] labelPositionTensors = new Tensor[MAX_LABELS];
        #endregion

        #region Initialization
        /// <summary>
        /// Initialize the label rendering system.
        /// Call this after SecureMR provider is created.
        /// </summary>
        public void Initialize(Provider provider)
        {
            this.provider = provider;
            CreateLabelPipeline();
            LogDebug("ObjectLabelRenderer initialized");
        }

        private void CreateLabelPipeline()
        {
            if (provider == null)
            {
                LogError("Provider is null, cannot create label pipeline");
                return;
            }

            LogDebug("Creating label rendering pipeline...");

            labelPipeline = provider.CreatePipeline();

            // Create tensors for each possible label
            for (int i = 0; i < MAX_LABELS; i++)
            {
                CreateLabelTensors(i);
            }

            LogDebug($"✓ Label pipeline created with {MAX_LABELS} label slots");
        }

        private void CreateLabelTensors(int index)
        {
            // Text content tensor (256 bytes per label)
            labelTextTensors[index] = provider.CreateTensor<byte, Scalar>(
                1,
                new TensorShape(new[] { 256 })
            );

            // Create text placeholder for pipeline
            labelTextPlaceholders[index] = labelPipeline.CreateTensorReference<byte, Scalar>(
                1,
                new TensorShape(new[] { 256 })
            );

            // Create RenderTextOperator for this label
            var textConfig = new RenderTextOperatorConfiguration(
                SecureMRFontTypeface.SansSerif,
                "en-US",
                1440,
                960
            );

            var textOp = labelPipeline.CreateOperator<RenderTextOperator>(textConfig);
            textOp.SetOperand("text", labelTextPlaceholders[index]);

            // Position tensor (will be updated per detection)
            labelPositionTensors[index] = labelPipeline.CreateTensor<float, Point>(
                2,
                new TensorShape(new[] { 1 })
            );
            textOp.SetOperand("position", labelPositionTensors[index]);

            // Color configuration
            var colors = labelPipeline.CreateTensor<byte, Unity.XR.PXR.SecureMR.Color>(
                4,
                new TensorShape(new[] { 2 })
            );
            
            byte[] colorData = new byte[]
            {
                (byte)(labelTextColor.r * 255),
                (byte)(labelTextColor.g * 255),
                (byte)(labelTextColor.b * 255),
                (byte)(labelTextColor.a * 255),
                (byte)(labelBackgroundColor.r * 255),
                (byte)(labelBackgroundColor.g * 255),
                (byte)(labelBackgroundColor.b * 255),
                (byte)(labelBackgroundColor.a * 255)
            };
            colors.Reset(colorData);
            textOp.SetOperand("colors", colors);

            // Texture ID
            var textureId = labelPipeline.CreateTensor<ushort, Scalar>(
                1,
                new TensorShape(new[] { 1 })
            );
            textureId.Reset(new ushort[] { (ushort)index }); // Unique texture per label
            textOp.SetOperand("texture_id", textureId);
        }
        #endregion

        #region Label Updates
        /// <summary>
        /// Update labels based on detection results.
        /// </summary>
        /// <param name="detections">List of detections with position and class info</param>
        public void UpdateLabels(List<DetectionResult> detections)
        {
            if (detections == null || labelPipeline == null) return;

            // Clear old labels
            activeLabelData.Clear();

            // Process up to MAX_LABELS detections
            int numLabels = Mathf.Min(detections.Count, MAX_LABELS);

            for (int i = 0; i < numLabels; i++)
            {
                var detection = detections[i];
                UpdateSingleLabel(i, detection);
            }

            // Clear unused label slots
            for (int i = numLabels; i < MAX_LABELS; i++)
            {
                ClearLabel(i);
            }

            // Run the label rendering pipeline with tensor mapping
            try
            {
                var mapping = new TensorMapping();
                for (int i = 0; i < MAX_LABELS; i++)
                {
                    mapping.Set(labelTextPlaceholders[i], labelTextTensors[i]);
                }
                labelPipeline.Execute(mapping);
                LogDebug($"Updated {numLabels} labels");
            }
            catch (Exception e)
            {
                LogError($"Failed to run label pipeline: {e.Message}");
            }
        }

        private void UpdateSingleLabel(int index, DetectionResult detection)
        {
            // Format label text
            string labelText = FormatLabel(detection);

            // Convert to UTF-8 bytes
            byte[] textBytes = System.Text.Encoding.UTF8.GetBytes(labelText);
            Array.Resize(ref textBytes, 256); // Pad to tensor size

            try
            {
                // Update text content
                labelTextTensors[index].Reset(textBytes);

                // Update position (place label above detection)
                // Normalized screen coordinates (0-1)
                float labelX = detection.centerX;
                float labelY = detection.centerY - (detection.height / 2) - 0.05f; // Above box

                labelPositionTensors[index].Reset(new float[] { labelX, labelY });

                activeLabelData.Add(new LabelData
                {
                    index = index,
                    text = labelText,
                    x = labelX,
                    y = labelY
                });

                LogDebug($"Label {index}: '{labelText}' at ({labelX:F2}, {labelY:F2})");
            }
            catch (Exception e)
            {
                LogError($"Failed to update label {index}: {e.Message}");
            }
        }

        private void ClearLabel(int index)
        {
            try
            {
                // Set empty text
                byte[] emptyBytes = new byte[256];
                labelTextTensors[index].Reset(emptyBytes);
            }
            catch (Exception e)
            {
                LogError($"Failed to clear label {index}: {e.Message}");
            }
        }

        private string FormatLabel(DetectionResult detection)
        {
            string className = classNames.ContainsKey(detection.classId)
                ? classNames[detection.classId]
                : $"Unknown ({detection.classId})";

            if (showConfidence)
            {
                return $"{className}\n{(detection.confidence * 100):F0}%";
            }
            else
            {
                return className;
            }
        }
        #endregion

        #region Cleanup
        private void OnDestroy()
        {
            LogDebug("Cleaning up ObjectLabelRenderer...");

            // Pipeline cleanup handled by SecureMR SDK

            LogDebug("✓ Cleanup complete");
        }
        #endregion

        #region Utilities
        private void LogDebug(string message)
        {
            if (debugLogging)
            {
                Debug.Log($"[ObjectLabelRenderer] {message}");
            }
        }

        private void LogError(string message)
        {
            Debug.LogError($"[ObjectLabelRenderer] {message}");
        }
        #endregion

        #region Helper Structures
        /// <summary>
        /// Detection result structure.
        /// Matches YOLO output format with normalized coordinates (0-1).
        /// </summary>
        [System.Serializable]
        public struct DetectionResult
        {
            public float centerX;    // Normalized center X (0-1)
            public float centerY;    // Normalized center Y (0-1)
            public float width;      // Normalized width (0-1)
            public float height;     // Normalized height (0-1)
            public float confidence; // Detection confidence (0-1)
            public int classId;      // COCO class ID

            public DetectionResult(float x, float y, float w, float h, int cls, float conf)
            {
                centerX = x;
                centerY = y;
                width = w;
                height = h;
                classId = cls;
                confidence = conf;
            }

            public override string ToString()
            {
                return $"Detection(class={classId}, conf={confidence:F2}, " +
                       $"pos=({centerX:F2},{centerY:F2}), size=({width:F2}x{height:F2}))";
            }
        }

        private struct LabelData
        {
            public int index;
            public string text;
            public float x;
            public float y;
        }
        #endregion

        #region Debug Visualization
        private void OnGUI()
        {
            if (!debugLogging || activeLabelData.Count == 0) return;

            GUIStyle style = new GUIStyle
            {
                fontSize = 14,
                padding = new RectOffset(5, 5, 5, 5)
            };
            style.normal.textColor = UnityEngine.Color.cyan;

            string info = $"Active Labels: {activeLabelData.Count}\n";
            for (int i = 0; i < activeLabelData.Count; i++)
            {
                var label = activeLabelData[i];
                info += $"{i}: {label.text} @ ({label.x:F2}, {label.y:F2})\n";
            }

            GUI.Label(new Rect(10, Screen.height - 200, 300, 200), info, style);
        }
        #endregion
    }

    #region Example Usage
    /// <summary>
    /// Example usage of ObjectLabelRenderer with YOLO detections.
    /// Add this to your main detection script.
    /// </summary>
    public class ObjectLabelRendererExample : MonoBehaviour
    {
        private ObjectLabelRenderer labelRenderer;
        private Provider provider;

        private void Start()
        {
            // Initialize SecureMR provider with VST resolution
            const int vstWidth = 1440;
            const int vstHeight = 960;
            provider = new Provider(vstWidth, vstHeight);

            // Get or add label renderer component
            labelRenderer = gameObject.AddComponent<ObjectLabelRenderer>();
            labelRenderer.Initialize(provider);
        }

        /// <summary>
        /// Call this when you have new detection results from YOLO.
        /// </summary>
        private void OnDetectionsReceived(List<ObjectLabelRenderer.DetectionResult> detections)
        {
            // Update labels with detection results
            labelRenderer.UpdateLabels(detections);
        }

        /// <summary>
        /// Example: Create mock detections for testing.
        /// </summary>
        private List<ObjectLabelRenderer.DetectionResult> CreateMockDetections()
        {
            return new List<ObjectLabelRenderer.DetectionResult>
            {
                // Bottle at left zone
                new ObjectLabelRenderer.DetectionResult(0.25f, 0.5f, 0.1f, 0.15f, 39, 0.92f),
                
                // Cup at right zone
                new ObjectLabelRenderer.DetectionResult(0.75f, 0.5f, 0.08f, 0.12f, 41, 0.88f),
                
                // Scissors at top zone
                new ObjectLabelRenderer.DetectionResult(0.5f, 0.25f, 0.12f, 0.1f, 76, 0.85f),
                
                // Book at bottom zone
                new ObjectLabelRenderer.DetectionResult(0.5f, 0.75f, 0.15f, 0.2f, 73, 0.91f)
            };
        }
    }
    #endregion
}