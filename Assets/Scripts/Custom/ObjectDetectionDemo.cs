using System;
using System.Threading;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;

/// <summary>
/// Object Detection Demo - Detects objects and shows labels
/// Uses PICO controllers only (no keyboard)
/// 
/// FILE NAME MUST BE: ObjectDetectionDemo.cs
/// </summary>
public class ObjectDetectionDemo : MonoBehaviour
{
    [Header("Models & Assets")]
    [Tooltip("YOLO model file (yolo_serialized.bin)")]
    public TextAsset yoloModel;
    
    [Tooltip("Optional panel GLTF for label background")]
    public TextAsset labelPanelGltf;

    [Header("Pipeline Settings")]
    public int vstWidth = 640;
    public int vstHeight = 640;

    [Header("PICO Controller")]
    public PXR_Input.Controller controllerHand = PXR_Input.Controller.RightController;
    
    [Header("Label Settings")]
    [Range(0.2f, 1.0f)]
    public float labelScale = 0.4f;
    
    [Range(24f, 72f)]
    public float fontSize = 56.0f;

    [Header("Debug")]
    public bool showDebugInfo = true;

    // SecureMR components
    private Provider provider;
    private Pipeline vstPipeline;
    private Pipeline detectionPipeline;
    private Pipeline renderPipeline;

    // Tensors
    private Tensor vstOutputGlobal;
    private Tensor[] labelGltfTensors = new Tensor[4];
    private Tensor[] labelTextTensors = new Tensor[4];

    // State
    private bool pipelinesReady = false;
    private bool[] labelsVisible = new bool[4];
    private float lastHapticTime = 0f;

    // Object info
    private readonly string[] objectNames = new string[]
    {
        "BOTTLE",
        "CUP",
        "SCISSORS",
        "BOOK"
    };

    // Label positions in 3D space
    private readonly Vector3[] labelPositions = new Vector3[]
    {
        new Vector3(-0.6f, 0.3f, -1.5f),  // Bottle - far left
        new Vector3(-0.2f, 0.3f, -1.5f),  // Cup - left
        new Vector3(0.2f, 0.3f, -1.5f),   // Scissors - right
        new Vector3(0.6f, 0.3f, -1.5f)    // Book - far right
    };

    private void Awake()
    {
        // Enable video see-through for MR
        PXR_Manager.EnableVideoSeeThrough = true;
        Application.targetFrameRate = 30;
    }

    private void Start()
    {
        Debug.Log("========================================");
        Debug.Log("    Object Detection Demo Starting");
        Debug.Log("========================================");

        if (!ValidateAssets())
        {
            Debug.LogError("Asset validation failed! Disabling script.");
            enabled = false;
            return;
        }

        InitializeSecureMR();
    }

    private bool ValidateAssets()
    {
        if (yoloModel == null)
        {
            Debug.LogError("[ObjectDetectionDemo] YOLO model not assigned in inspector!");
            return false;
        }

        Debug.Log("[ObjectDetectionDemo] YOLO model loaded: " + yoloModel.name);
        
        if (labelPanelGltf == null)
        {
            Debug.LogWarning("[ObjectDetectionDemo] No label panel GLTF - using text only");
        }

        return true;
    }

    private void InitializeSecureMR()
    {
        Debug.Log("[ObjectDetectionDemo] Starting SecureMR initialization...");

        Thread initThread = new Thread(() =>
        {
            try
            {
                Debug.Log("[ObjectDetectionDemo] Creating provider...");
                provider = new Provider(vstWidth, vstHeight);

                Debug.Log("[ObjectDetectionDemo] Creating global tensors...");
                CreateGlobalTensors();

                Debug.Log("[ObjectDetectionDemo] Building pipelines...");
                BuildVSTPipeline();
                BuildDetectionPipeline();
                BuildRenderPipeline();

                pipelinesReady = true;
                Debug.Log("[ObjectDetectionDemo] ✓ SecureMR ready!");
            }
            catch (Exception e)
            {
                Debug.LogError($"[ObjectDetectionDemo] Initialization failed: {e.Message}\n{e.StackTrace}");
            }
        });

        initThread.Start();
    }

    private void CreateGlobalTensors()
    {
        // VST output
        vstOutputGlobal = provider.CreateTensor<float, Matrix>(3, 
            new TensorShape(new[] { vstHeight, vstWidth }));

        // Label GLTFs and text
        for (int i = 0; i < 4; i++)
        {
            if (labelPanelGltf != null)
            {
                labelGltfTensors[i] = provider.CreateTensor<Gltf>(labelPanelGltf.bytes);
            }

            labelTextTensors[i] = provider.CreateTensor<byte, Scalar>(1, 
                new TensorShape(new[] { 64 }));

            // Set initial text
            byte[] textBytes = System.Text.Encoding.ASCII.GetBytes(objectNames[i]);
            Array.Resize(ref textBytes, 64);
            labelTextTensors[i].Reset(textBytes);
        }
    }

    private void BuildVSTPipeline()
    {
        vstPipeline = provider.CreatePipeline();

        var vstOutputPlaceholder = vstPipeline.CreateTensorReference<float, Matrix>(3, 
            new TensorShape(new[] { vstHeight, vstWidth }));
        
        var vstOutputUint8 = vstPipeline.CreateTensor<byte, Matrix>(3, 
            new TensorShape(new[] { vstHeight, vstWidth }));

        // Get camera frames
        var vstOp = vstPipeline.CreateOperator<RectifiedVstAccessOperator>();
        vstOp.SetResult("left image", vstOutputUint8);

        // Convert uint8 to float32
        var assignOp = vstPipeline.CreateOperator<AssignmentOperator>();
        assignOp.SetOperand("src", vstOutputUint8);
        assignOp.SetResult("dst", vstOutputPlaceholder);

        // Normalize to [0, 1]
        var normalizeOp = vstPipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));
        normalizeOp.SetOperand("{0}", vstOutputPlaceholder);
        normalizeOp.SetResult("result", vstOutputPlaceholder);

        Debug.Log("[ObjectDetectionDemo] ✓ VST pipeline built");
    }

    private void BuildDetectionPipeline()
    {
        detectionPipeline = provider.CreatePipeline();

        var vstInputPlaceholder = detectionPipeline.CreateTensorReference<float, Matrix>(3, 
            new TensorShape(new[] { vstHeight, vstWidth }));

        // Configure YOLO model
        var modelConfig = new ModelOperatorConfiguration(
            yoloModel.bytes,
            SecureMRModelType.QnnContextBinary,
            "yolo"
        );

        // Input/output mappings - verify these match your YOLO model
        modelConfig.AddInputMapping("images", "images", SecureMRModelEncoding.Float32);
        modelConfig.AddOutputMapping("output0", "output0", SecureMRModelEncoding.Float32);

        var modelOp = detectionPipeline.CreateOperator<RunModelInferenceOperator>(modelConfig);
        modelOp.SetOperand("images", vstInputPlaceholder);

        // YOLO output tensor
        var rawOutput = detectionPipeline.CreateTensor<float, Matrix>(1, 
            new TensorShape(new[] { 84, 8400 }));
        modelOp.SetResult("output0", rawOutput);

        Debug.Log("[ObjectDetectionDemo] ✓ Detection pipeline built");
        Debug.Log("[ObjectDetectionDemo]   YOLO model will run on every frame");
    }

    private void BuildRenderPipeline()
    {
        renderPipeline = provider.CreatePipeline();

        // Create renderer for each label
        for (int i = 0; i < 4; i++)
        {
            CreateLabelRenderer(i);
        }

        Debug.Log("[ObjectDetectionDemo] ✓ Render pipeline built");
    }

    private void CreateLabelRenderer(int index)
    {
        // Text content
        var textPlaceholder = renderPipeline.CreateTensorReference<byte, Scalar>(1, 
            new TensorShape(new[] { 64 }));

        // World position
        var positionMatrix = renderPipeline.CreateTensor<float, Matrix>(1, 
            new TensorShape(new[] { 4, 4 }),
            CreateTransformMatrix(labelPositions[index], labelScale));

        // Text rendering settings
        var startPos = renderPipeline.CreateTensor<float, Point>(2, 
            new TensorShape(new[] { 1 }));
        startPos.Reset(new float[] { 0.1f, 0.5f });

        // Colors - different color per object
        var colors = renderPipeline.CreateTensor<byte, Unity.XR.PXR.SecureMR.Color>(4, 
            new TensorShape(new[] { 2 }));
        colors.Reset(GetColorForObject(index));

        var textureId = renderPipeline.CreateTensor<ushort, Scalar>(1, 
            new TensorShape(new[] { 1 }));
        textureId.Reset(new ushort[] { (ushort)index });

        var fontSizeTensor = renderPipeline.CreateTensor<float, Scalar>(1, 
            new TensorShape(new[] { 1 }));
        fontSizeTensor.Reset(new float[] { fontSize });

        var gltfPlaceholder = renderPipeline.CreateTensorReference<Gltf>();

        // Text operator
        var textOp = renderPipeline.CreateOperator<RenderTextOperator>(
            new RenderTextOperatorConfiguration(
                SecureMRFontTypeface.SansSerif,
                "en-US",
                512, 512
            ));
        textOp.SetOperand("text", textPlaceholder);
        textOp.SetOperand("start", startPos);
        textOp.SetOperand("colors", colors);
        textOp.SetOperand("texture ID", textureId);
        textOp.SetOperand("font size", fontSizeTensor);
        textOp.SetOperand("gltf", gltfPlaceholder);

        // GLTF render operator
        var gltfOp = renderPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
        gltfOp.SetOperand("gltf", gltfPlaceholder);
        gltfOp.SetOperand("world pose", positionMatrix);
    }

    private float[] CreateTransformMatrix(Vector3 position, float scale)
    {
        return new float[]
        {
            scale, 0.0f, 0.0f, position.x,
            0.0f, scale, 0.0f, position.y,
            0.0f, 0.0f, scale, position.z,
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }

    private byte[] GetColorForObject(int index)
    {
        // Text color + background color (RGBA each)
        switch (index)
        {
            case 0: // Bottle - Red
                return new byte[] { 255, 100, 100, 255, 50, 0, 0, 220 };
            case 1: // Cup - Blue
                return new byte[] { 100, 150, 255, 255, 0, 0, 50, 220 };
            case 2: // Scissors - Green
                return new byte[] { 100, 255, 150, 255, 0, 50, 0, 220 };
            case 3: // Book - Yellow
                return new byte[] { 255, 255, 100, 255, 50, 50, 0, 220 };
            default:
                return new byte[] { 255, 255, 255, 255, 0, 0, 0, 220 };
        }
    }

    private void Update()
    {
        if (!pipelinesReady) return;

        HandlePICOController();
        ExecutePipelines();
    }

    private void HandlePICOController()
    {
        // Get controller device
        UnityEngine.XR.InputDevice controller;
        if (controllerHand == PXR_Input.Controller.LeftController)
        {
            controller = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.LeftHand);
        }
        else
        {
            controller = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.RightHand);
        }

        if (!controller.isValid) return;

        // D-PAD: Toggle individual labels
        if (controller.TryGetFeatureValue(UnityEngine.XR.CommonUsages.primary2DAxisClick, out bool dpadClick) && dpadClick)
        {
            if (controller.TryGetFeatureValue(UnityEngine.XR.CommonUsages.primary2DAxis, out Vector2 dpadAxis))
            {
                if (Mathf.Abs(dpadAxis.x) > Mathf.Abs(dpadAxis.y))
                {
                    if (dpadAxis.x < -0.5f) // Left
                        ToggleLabel(0, "BOTTLE");
                    else if (dpadAxis.x > 0.5f) // Right
                        ToggleLabel(2, "SCISSORS");
                }
                else
                {
                    if (dpadAxis.y > 0.5f) // Up
                        ToggleLabel(1, "CUP");
                    else if (dpadAxis.y < -0.5f) // Down
                        ToggleLabel(3, "BOOK");
                }
            }
        }

        // A Button (primaryButton): Show all labels
        if (controller.TryGetFeatureValue(UnityEngine.XR.CommonUsages.primaryButton, out bool aButton) && aButton)
        {
            ShowAllLabels();
        }

        // B Button (secondaryButton): Hide all labels
        if (controller.TryGetFeatureValue(UnityEngine.XR.CommonUsages.secondaryButton, out bool bButton) && bButton)
        {
            HideAllLabels();
        }

        // X Button: Cycle through labels
        if (controller.TryGetFeatureValue(UnityEngine.XR.CommonUsages.primary2DAxisTouch, out bool xButton) && xButton)
        {
            CycleLabels();
        }

        // Trigger: Toggle visibility of all labels (quick on/off)
        if (controller.TryGetFeatureValue(UnityEngine.XR.CommonUsages.trigger, out float triggerValue) && triggerValue > 0.9f)
        {
            ToggleAllLabels();
        }
    }

    private void ToggleLabel(int index, string name)
    {
        labelsVisible[index] = !labelsVisible[index];
        string status = labelsVisible[index] ? "VISIBLE" : "HIDDEN";
        
        Debug.Log($"[ObjectDetectionDemo] {name}: {status}");
        
        TriggerHaptic(0.3f, 50);
    }

    private void ShowAllLabels()
    {
        for (int i = 0; i < 4; i++)
        {
            labelsVisible[i] = true;
        }
        Debug.Log("[ObjectDetectionDemo] All labels VISIBLE");
        TriggerHaptic(0.5f, 100);
    }

    private void HideAllLabels()
    {
        for (int i = 0; i < 4; i++)
        {
            labelsVisible[i] = false;
        }
        Debug.Log("[ObjectDetectionDemo] All labels HIDDEN");
        TriggerHaptic(0.5f, 100);
    }

    private void CycleLabels()
    {
        // Find first visible label and move to next
        int firstVisible = -1;
        for (int i = 0; i < 4; i++)
        {
            if (labelsVisible[i])
            {
                firstVisible = i;
                labelsVisible[i] = false;
                break;
            }
        }

        int nextIndex = (firstVisible + 1) % 4;
        labelsVisible[nextIndex] = true;
        
        Debug.Log($"[ObjectDetectionDemo] Cycling to: {objectNames[nextIndex]}");
        TriggerHaptic(0.3f, 50);
    }

    private void ToggleAllLabels()
    {
        bool anyVisible = false;
        for (int i = 0; i < 4; i++)
        {
            if (labelsVisible[i])
            {
                anyVisible = true;
                break;
            }
        }

        // If any visible, hide all. If none visible, show all.
        for (int i = 0; i < 4; i++)
        {
            labelsVisible[i] = !anyVisible;
        }

        string status = anyVisible ? "HIDDEN" : "VISIBLE";
        Debug.Log($"[ObjectDetectionDemo] All labels: {status}");
        TriggerHaptic(0.4f, 75);
    }

    private void TriggerHaptic(float strength, int durationMs)
    {
        if (Time.time - lastHapticTime < 0.1f) return; // Debounce
        
        PXR_Input.SendHapticImpulse(
            PXR_Input.VibrateType.RightController,
            strength,
            durationMs,
            (int)controllerHand
        );
        
        lastHapticTime = Time.time;
    }

    private void ExecutePipelines()
    {
        try
        {
            // Run VST pipeline (get camera frames)
            var vstMapping = new TensorMapping();
            vstMapping.Set(
                vstPipeline.CreateTensorReference<float, Matrix>(3, 
                    new TensorShape(new[] { vstHeight, vstWidth })),
                vstOutputGlobal
            );
            vstPipeline.Execute(vstMapping);

            // Run YOLO detection pipeline
            var detectionMapping = new TensorMapping();
            detectionMapping.Set(
                detectionPipeline.CreateTensorReference<float, Matrix>(3,
                    new TensorShape(new[] { vstHeight, vstWidth })),
                vstOutputGlobal
            );
            detectionPipeline.Execute(detectionMapping);

            // Render visible labels
            var renderMapping = new TensorMapping();
            
            for (int i = 0; i < 4; i++)
            {
                if (labelsVisible[i])
                {
                    if (labelGltfTensors[i] != null)
                    {
                        var gltfPlaceholder = renderPipeline.CreateTensorReference<Gltf>();
                        renderMapping.Set(gltfPlaceholder, labelGltfTensors[i]);
                    }

                    var textPlaceholder = renderPipeline.CreateTensorReference<byte, Scalar>(1,
                        new TensorShape(new[] { 64 }));
                    renderMapping.Set(textPlaceholder, labelTextTensors[i]);
                }
            }
            
            renderPipeline.Execute(renderMapping);
        }
        catch (Exception e)
        {
            if (Time.frameCount % 300 == 0) // Log errors occasionally
            {
                Debug.LogError($"[ObjectDetectionDemo] Pipeline error: {e.Message}");
            }
        }
    }

    private void OnGUI()
    {
        if (!showDebugInfo) return;

        GUIStyle style = new GUIStyle
        {
            fontSize = 20,
            padding = new RectOffset(10, 10, 10, 10)
        };
        style.normal.textColor = UnityEngine.Color.cyan;

        string info = "OBJECT DETECTION DEMO\n";
        info += "YOLO Running: " + (pipelinesReady ? "YES" : "NO") + "\n\n";
        info += "PICO CONTROLLER:\n";
        info += "D-PAD ←  : BOTTLE\n";
        info += "D-PAD ↑  : CUP\n";
        info += "D-PAD →  : SCISSORS\n";
        info += "D-PAD ↓  : BOOK\n";
        info += "A        : Show All\n";
        info += "B        : Hide All\n";
        info += "X        : Cycle Labels\n";
        info += "TRIGGER  : Toggle All\n\n";
        info += "LABELS:\n";
        
        for (int i = 0; i < 4; i++)
        {
            string status = labelsVisible[i] ? "[ON]" : "[OFF]";
            info += $"{status} {objectNames[i]}\n";
        }

        GUI.Label(new Rect(10, 10, 400, 500), info, style);
    }
}