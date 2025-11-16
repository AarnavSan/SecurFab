using UnityEngine;
using SecureFab.Training;

namespace PicoXR.SecureMR.Demo
{
    /// <summary>
    /// Simulates object detection for testing the training system
    /// without requiring YOLO to be fully operational.
    /// 
    /// USE THIS FOR RAPID PROTOTYPING during the hackathon!
    /// </summary>
    public class DetectionSimulator : MonoBehaviour
    {
        [Header("Simulation Mode")]
        [Tooltip("Enable keyboard-based object placement simulation")]
        public bool enableKeyboardSimulation = true;

        [Tooltip("Enable automatic step-through simulation")]
        public bool enableAutoStepThrough = false;

        [Range(1f, 10f)]
        public float autoStepDelay = 3f;

        [Header("References")]
        public StepManager stepManager;

        [Header("Debug")]
        public bool debugLogging = true;

        private ExpectedConfig simulatedConfig = new ExpectedConfig();
        private float autoStepTimer = 0f;

        private void Update()
        {
            if (stepManager == null || !stepManager.IsInitialized) return;

            if (enableKeyboardSimulation)
            {
                HandleKeyboardSimulation();
            }

            if (enableAutoStepThrough)
            {
                HandleAutoStepThrough();
            }
        }

        private void HandleKeyboardSimulation()
        {
            // ZONE SELECTION + OBJECT PLACEMENT
            // Hold zone key (Q/W/E/R) + object key (1/2/3/4)

            bool leftZone = Input.GetKey(KeyCode.Q);
            bool rightZone = Input.GetKey(KeyCode.W);
            bool topZone = Input.GetKey(KeyCode.E);
            bool bottomZone = Input.GetKey(KeyCode.R);

            // Object placement
            if (Input.GetKeyDown(KeyCode.Alpha1)) // Bottle
            {
                PlaceObject("bottle", leftZone, rightZone, topZone, bottomZone);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha2)) // Cup
            {
                PlaceObject("cup", leftZone, rightZone, topZone, bottomZone);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha3)) // Scissors
            {
                PlaceObject("scissors", leftZone, rightZone, topZone, bottomZone);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha4)) // Book
            {
                PlaceObject("book", leftZone, rightZone, topZone, bottomZone);
            }

            // Clear zones
            if (Input.GetKeyDown(KeyCode.C))
            {
                ClearZone(leftZone, rightZone, topZone, bottomZone);
            }

            // Clear all
            if (Input.GetKeyDown(KeyCode.X))
            {
                ClearAllZones();
            }

            // Validate current configuration
            if (Input.GetKeyDown(KeyCode.V))
            {
                ValidateConfiguration();
            }

            // Auto-fill correct configuration for current step
            if (Input.GetKeyDown(KeyCode.Space))
            {
                AutoFillCorrectConfiguration();
            }
        }

        private void PlaceObject(string objectName, bool left, bool right, bool top, bool bottom)
        {
            if (left)
            {
                simulatedConfig.left = objectName;
                SecureFabLogger.Log("Simulator", $"Placed {objectName} in LEFT zone");
            }
            else if (right)
            {
                simulatedConfig.right = objectName;
                SecureFabLogger.Log("Simulator", $"Placed {objectName} in RIGHT zone");
            }
            else if (top)
            {
                simulatedConfig.top = objectName;
                SecureFabLogger.Log("Simulator", $"Placed {objectName} in TOP zone");
            }
            else if (bottom)
            {
                simulatedConfig.bottom = objectName;
                SecureFabLogger.Log("Simulator", $"Placed {objectName} in BOTTOM zone");
            }
            else
            {
                SecureFabLogger.LogWarning("Simulator", "No zone key held! Hold Q/W/E/R and press object key.");
            }

            LogCurrentConfiguration();
        }

        private void ClearZone(bool left, bool right, bool top, bool bottom)
        {
            if (left)
            {
                simulatedConfig.left = null;
                SecureFabLogger.Log("Simulator", "Cleared LEFT zone");
            }
            else if (right)
            {
                simulatedConfig.right = null;
                SecureFabLogger.Log("Simulator", "Cleared RIGHT zone");
            }
            else if (top)
            {
                simulatedConfig.top = null;
                SecureFabLogger.Log("Simulator", "Cleared TOP zone");
            }
            else if (bottom)
            {
                simulatedConfig.bottom = null;
                SecureFabLogger.Log("Simulator", "Cleared BOTTOM zone");
            }
            else
            {
                SecureFabLogger.LogWarning("Simulator", "No zone key held! Hold Q/W/E/R and press C to clear.");
            }

            LogCurrentConfiguration();
        }

        private void ClearAllZones()
        {
            simulatedConfig = new ExpectedConfig();
            SecureFabLogger.Log("Simulator", "Cleared ALL zones");
            LogCurrentConfiguration();
        }

        private void ValidateConfiguration()
        {
            if (stepManager != null)
            {
                bool isValid = stepManager.ValidateConfiguration(simulatedConfig);
                SecureFabLogger.Log("Simulator", $"Manual validation: {(isValid ? "✓ PASS" : "✗ FAIL")}");
            }
        }

        private void AutoFillCorrectConfiguration()
        {
            if (stepManager == null || stepManager.CurrentStep == null) return;

            var expected = stepManager.CurrentStep.expected_config;
            simulatedConfig = new ExpectedConfig
            {
                left = expected.left,
                right = expected.right,
                top = expected.top,
                bottom = expected.bottom
            };

            SecureFabLogger.Log("Simulator", "Auto-filled correct configuration for current step:");
            LogCurrentConfiguration();

            // Auto-validate
            ValidateConfiguration();
        }

        private void HandleAutoStepThrough()
        {
            autoStepTimer += Time.deltaTime;

            if (autoStepTimer >= autoStepDelay)
            {
                autoStepTimer = 0f;

                // Auto-fill and validate
                AutoFillCorrectConfiguration();

                // Small delay before next step
                autoStepDelay = Random.Range(2f, 4f);
            }
        }

        private void LogCurrentConfiguration()
        {
            SecureFabLogger.LogConfig("CurrentSimulated", simulatedConfig);
        }

        private void OnGUI()
        {
            if (!debugLogging) return;

            GUIStyle style = new GUIStyle
            {
                fontSize = 16,
                padding = new RectOffset(10, 10, 10, 10)
            };
            style.normal.textColor = Color.cyan;

            string helpText = "DETECTION SIMULATOR CONTROLS\n\n";
            helpText += "OBJECT PLACEMENT:\n";
            helpText += "  Hold Q (left) / W (right) / E (top) / R (bottom)\n";
            helpText += "  + Press 1 (bottle) / 2 (cup) / 3 (scissors) / 4 (book)\n\n";
            helpText += "ZONE MANAGEMENT:\n";
            helpText += "  Hold zone key + C: Clear zone\n";
            helpText += "  X: Clear all zones\n\n";
            helpText += "VALIDATION:\n";
            helpText += "  V: Validate current config\n";
            helpText += "  SPACE: Auto-fill correct config\n\n";
            helpText += "CURRENT SIMULATED CONFIG:\n";
            helpText += $"{simulatedConfig}";

            GUI.Label(new Rect(Screen.width - 510, 10, 500, 400), helpText, style);
        }
    }
}