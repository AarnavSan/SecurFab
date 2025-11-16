using UnityEngine;
using Unity.XR.PXR;
using SecureFab.Training;

namespace PicoXR.SecureMR.Demo
{
    /// <summary>
    /// Master integration script that coordinates all SecureFab training components.
    /// Acts as the "brain" of the system, managing interactions between:
    /// - StepManager (training flow)
    /// - SecureMRObjectDetector (vision system)
    /// - PicoControllerInput (user input)
    /// - UI components (visual feedback)
    /// </summary>
    public class SecureFabMasterController : MonoBehaviour
    {
        [Header("Core Components")]
        [Tooltip("Manages training step progression")]
        public StepManager stepManager;

        [Tooltip("Handles object detection via SecureMR")]
        public SecureMRObjectDetector objectDetector;

        [Tooltip("Manages PICO controller input")]
        public PicoControllerInput controllerInput;

        [Tooltip("Optional: Detection simulator for testing")]
        public DetectionSimulator simulator;

        [Header("System Configuration")]
        [Tooltip("Mode selection")]
        public OperationMode mode = OperationMode.AutoDetection;

        [Tooltip("Enable comprehensive logging")]
        public bool masterDebugLogging = true;

        [Header("Validation Settings")]
        [Range(0.5f, 5f)]
        [Tooltip("Cooldown between validation attempts (seconds)")]
        public float validationCooldown = 1.5f;

        [Range(1, 10)]
        [Tooltip("Number of consistent detections required before validation")]
        public int validationConsistencyRequired = 3;

        [Header("Feedback Settings")]
        public bool enableSuccessCelebration = true;
        public bool enableErrorWarnings = true;
        public float successHapticDuration = 100f;
        public float errorHapticDuration = 200f;

        // Operation modes
        public enum OperationMode
        {
            AutoDetection,      // Full YOLO detection → auto-progression
            ManualProgression,  // Detection for feedback, manual step advance
            SimulatorOnly       // Use DetectionSimulator for testing
        }

        // State tracking
        private bool isInitialized = false;
        private float lastValidationTime = 0f;
        private int consecutiveCorrectDetections = 0;
        private int consecutiveIncorrectDetections = 0;

        #region Unity Lifecycle
        private void Awake()
        {
            ValidateConfiguration();
        }

        private void Start()
        {
            InitializeSystem();
        }

        private void Update()
        {
            if (!isInitialized) return;

            // Mode-specific update logic
            switch (mode)
            {
                case OperationMode.AutoDetection:
                    UpdateAutoDetectionMode();
                    break;

                case OperationMode.ManualProgression:
                    UpdateManualMode();
                    break;

                case OperationMode.SimulatorOnly:
                    // Simulator handles its own updates
                    break;
            }
        }
        #endregion

        #region Initialization
        private bool ValidateConfiguration()
        {
            bool isValid = true;

            if (stepManager == null)
            {
                SecureFabLogger.LogError("MasterController", "StepManager reference missing!");
                isValid = false;
            }

            if (mode == OperationMode.AutoDetection || mode == OperationMode.ManualProgression)
            {
                if (objectDetector == null)
                {
                    SecureFabLogger.LogError("MasterController", "ObjectDetector required for selected mode!");
                    isValid = false;
                }
            }

            if (mode == OperationMode.SimulatorOnly && simulator == null)
            {
                SecureFabLogger.LogError("MasterController", "DetectionSimulator required for simulator mode!");
                isValid = false;
            }

            return isValid;
        }

        private void InitializeSystem()
        {
            SecureFabLogger.Log("MasterController", "╔═══════════════════════════════════════════╗");
            SecureFabLogger.Log("MasterController", "║   SecureFab Master Controller Starting   ║");
            SecureFabLogger.Log("MasterController", "╚═══════════════════════════════════════════╝");

            // Subscribe to events
            if (stepManager != null)
            {
                stepManager.onStepChanged.AddListener(OnStepChanged);
                stepManager.onConfigurationValidated.AddListener(OnConfigurationValidated);
                stepManager.onProcedureComplete.AddListener(OnProcedureComplete);
            }

            // Configure components based on mode
            ConfigureForMode(mode);

            isInitialized = true;

            SecureFabLogger.Log("MasterController", $"System initialized in {mode} mode");
            SecureFabLogger.Log("MasterController", $"Current step: {stepManager.CurrentStep.title}");
        }

        private void ConfigureForMode(OperationMode selectedMode)
        {
            switch (selectedMode)
            {
                case OperationMode.AutoDetection:
                    if (stepManager != null) stepManager.enableAutoProgress = true;
                    if (simulator != null) simulator.enabled = false;
                    SecureFabLogger.Log("MasterController", "Configured for AUTO-DETECTION mode");
                    break;

                case OperationMode.ManualProgression:
                    if (stepManager != null) stepManager.enableAutoProgress = false;
                    if (simulator != null) simulator.enabled = false;
                    SecureFabLogger.Log("MasterController", "Configured for MANUAL progression mode");
                    break;

                case OperationMode.SimulatorOnly:
                    if (stepManager != null) stepManager.enableAutoProgress = true;
                    if (simulator != null) simulator.enabled = true;
                    if (objectDetector != null) objectDetector.enabled = false;
                    SecureFabLogger.Log("MasterController", "Configured for SIMULATOR-ONLY mode");
                    break;
            }
        }
        #endregion

        #region Mode-Specific Updates
        private void UpdateAutoDetectionMode()
        {
            // In auto-detection mode:
            // - ObjectDetector continuously validates configuration
            // - On correct config, StepManager auto-advances
            // - Master controller provides additional feedback

            // Cooldown logic handled by ObjectDetector
            // We just coordinate the overall experience here
        }

        private void UpdateManualMode()
        {
            // In manual mode:
            // - ObjectDetector provides feedback
            // - User manually advances with controller
            // - We provide haptic/visual feedback only
        }
        #endregion

        #region Event Handlers
        private void OnStepChanged(Step newStep)
        {
            SecureFabLogger.Log("MasterController", "═══════════════════════════════════════");
            SecureFabLogger.Log("MasterController", $"STEP TRANSITION: {newStep.title}");
            SecureFabLogger.Log("MasterController", $"Progress: {stepManager.GetProgressString()}");
            SecureFabLogger.Log("MasterController", "═══════════════════════════════════════");

            // Reset validation counters
            consecutiveCorrectDetections = 0;
            consecutiveIncorrectDetections = 0;
            lastValidationTime = 0f;

            // Provide feedback for step change
            if (enableSuccessCelebration && controllerInput != null)
            {
                // Brief success haptic
                PXR_Input.SendHapticImpulse(
                    PXR_Input.VibrateType.BothController,
                    0.3f,
                    (int)successHapticDuration,
                    (int)controllerInput.controllerHand
                );
            }
        }

        private void OnConfigurationValidated(bool isValid)
        {
            if (isValid)
            {
                consecutiveCorrectDetections++;
                consecutiveIncorrectDetections = 0;

                SecureFabLogger.Log("MasterController", 
                    $"✓ Config VALID ({consecutiveCorrectDetections}/{validationConsistencyRequired} required)");

                // Celebrate success
                if (enableSuccessCelebration && consecutiveCorrectDetections >= validationConsistencyRequired)
                {
                    CelebrateSuccess();
                }
            }
            else
            {
                consecutiveIncorrectDetections++;
                consecutiveCorrectDetections = 0;

                SecureFabLogger.Log("MasterController", 
                    $"✗ Config INVALID (attempt {consecutiveIncorrectDetections})");

                // Warn about error
                if (enableErrorWarnings)
                {
                    WarnIncorrectConfiguration();
                }
            }
        }

        private void OnProcedureComplete()
        {
            SecureFabLogger.Log("MasterController", "╔═══════════════════════════════════════════╗");
            SecureFabLogger.Log("MasterController", "║   🎉 TRAINING COMPLETE! 🎉                ║");
            SecureFabLogger.Log("MasterController", "╚═══════════════════════════════════════════╝");

            // Celebration haptics
            if (controllerInput != null)
            {
                // Double pulse
                PXR_Input.SendHapticImpulse(
                    PXR_Input.VibrateType.BothController,
                    0.8f,
                    150,
                    (int)controllerInput.controllerHand
                );

                Invoke(nameof(SecondCelebrationPulse), 0.2f);
            }
        }

        private void SecondCelebrationPulse()
        {
            if (controllerInput != null)
            {
                PXR_Input.SendHapticImpulse(
                    PXR_Input.VibrateType.BothController,
                    0.8f,
                    150,
                    (int)controllerInput.controllerHand
                );
            }
        }
        #endregion

        #region Feedback Methods
        private void CelebrateSuccess()
        {
            if (controllerInput == null) return;

            // Success pattern: short, pleasant pulse
            PXR_Input.SendHapticImpulse(
                PXR_Input.VibrateType.BothController,
                0.5f,
                (int)successHapticDuration,
                (int)controllerInput.controllerHand
            );

            if (masterDebugLogging)
            {
                SecureFabLogger.Log("MasterController", "✓ Success feedback triggered");
            }
        }

        private void WarnIncorrectConfiguration()
        {
            if (controllerInput == null) return;

            // Error pattern: longer, stronger pulse
            PXR_Input.SendHapticImpulse(
                PXR_Input.VibrateType.BothController,
                0.7f,
                (int)errorHapticDuration,
                (int)controllerInput.controllerHand
            );

            if (masterDebugLogging)
            {
                SecureFabLogger.Log("MasterController", "⚠️ Error warning triggered");
            }
        }
        #endregion

        #region Public API
        /// <summary>
        /// Manually trigger a configuration check.
        /// Useful for testing or external triggers.
        /// </summary>
        public void TriggerManualValidation()
        {
            if (stepManager == null || objectDetector == null) return;

            ExpectedConfig currentDetected = objectDetector.CurrentDetectedConfiguration;
            stepManager.ValidateConfiguration(currentDetected);
        }

        /// <summary>
        /// Reset the training to the first step.
        /// </summary>
        public void ResetTraining()
        {
            if (stepManager != null)
            {
                stepManager.ResetToFirstStep();
                consecutiveCorrectDetections = 0;
                consecutiveIncorrectDetections = 0;

                SecureFabLogger.Log("MasterController", "Training reset to first step");
            }
        }

        /// <summary>
        /// Change operation mode at runtime.
        /// </summary>
        public void SetOperationMode(OperationMode newMode)
        {
            if (newMode == mode) return;

            mode = newMode;
            ConfigureForMode(mode);

            SecureFabLogger.Log("MasterController", $"Mode changed to: {mode}");
        }

        /// <summary>
        /// Get system status summary.
        /// </summary>
        public string GetSystemStatus()
        {
            if (!isInitialized)
                return "System not initialized";

            string status = $"Mode: {mode}\n";
            status += $"Step: {stepManager.GetProgressString()}\n";
            status += $"Consecutive Correct: {consecutiveCorrectDetections}\n";
            status += $"Consecutive Incorrect: {consecutiveIncorrectDetections}";

            return status;
        }
        #endregion

        #region Debug UI
        private void OnGUI()
        {
            if (!masterDebugLogging) return;

            GUIStyle titleStyle = new GUIStyle
            {
                fontSize = 20,
                fontStyle = FontStyle.Bold,
                padding = new RectOffset(10, 10, 10, 10)
            };
            titleStyle.normal.textColor = Color.yellow;

            GUIStyle bodyStyle = new GUIStyle
            {
                fontSize = 16,
                padding = new RectOffset(10, 10, 5, 5)
            };
            bodyStyle.normal.textColor = Color.white;

            string title = "SECUREFAB MASTER CONTROLLER\n";
            string status = GetSystemStatus();

            GUI.Label(new Rect(Screen.width - 510, 10, 500, 50), title, titleStyle);
            GUI.Label(new Rect(Screen.width - 510, 60, 500, 150), status, bodyStyle);
        }
        #endregion

        #region Cleanup
        private void OnDestroy()
        {
            if (stepManager != null)
            {
                stepManager.onStepChanged.RemoveListener(OnStepChanged);
                stepManager.onConfigurationValidated.RemoveListener(OnConfigurationValidated);
                stepManager.onProcedureComplete.RemoveListener(OnProcedureComplete);
            }
        }
        #endregion
    }
}