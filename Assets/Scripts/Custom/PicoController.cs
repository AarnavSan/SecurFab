using UnityEngine;
using Unity.XR.PXR;
using UnityEngine.XR;
using System.Collections.Generic;

namespace SecureFab.Training
{
    /// <summary>
    /// Handles PICO XR controller input for the SecureFab training demo.
    /// Uses Unity XR Input System which is the recommended approach for PICO Unity Integration SDK.
    /// </summary>
    public class PicoControllerInput : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("Reference to the StepManager in the scene")]
        public StepManager stepManager;

        [Header("Controller Selection")]
        [Tooltip("Which hand to use for controls")]
        public PXR_Input.Controller controllerHand = PXR_Input.Controller.RightController;

        [Header("Haptic Settings")]
        [Tooltip("Enable haptic feedback on button press")]
        public bool enableHaptics = true;

        [Tooltip("Haptic feedback strength (0-1)")]
        [Range(0f, 1f)]
        public float hapticStrength = 0.5f;

        [Tooltip("Haptic feedback duration in milliseconds")]
        [Range(10, 500)]
        public int hapticDurationMs = 50;

        [Header("Thumbstick Settings")]
        [Tooltip("Thumbstick threshold for directional input")]
        [Range(0.3f, 0.9f)]
        public float thumbstickThreshold = 0.7f;

        [Header("Debug")]
        public bool debugLogging = true;

        // Unity XR Input devices
        private InputDevice leftController;
        private InputDevice rightController;
        private InputDevice activeController;

        // Jump target tracking
        private int jumpToStepIndex = 0;
        private float lastThumbstickTime = 0f;
        private float thumbstickCooldown = 0.3f;

        // Track button states for "down" detection
        private bool lastPrimaryButton = false;
        private bool lastSecondaryButton = false;
        private bool lastGripButton = false;
        private bool lastTriggerButton = false;

        private void Start()
        {
            InitializeControllers();
        }

        private void InitializeControllers()
        {
            // Get XR input devices
            var leftDevices = new List<InputDevice>();
            var rightDevices = new List<InputDevice>();

            InputDevices.GetDevicesAtXRNode(XRNode.LeftHand, leftDevices);
            InputDevices.GetDevicesAtXRNode(XRNode.RightHand, rightDevices);

            if (leftDevices.Count > 0)
                leftController = leftDevices[0];

            if (rightDevices.Count > 0)
                rightController = rightDevices[0];

            // Set active controller based on selection
            UpdateActiveController();

            if (debugLogging)
            {
                string controllerName = controllerHand == PXR_Input.Controller.LeftController ? "Left" : "Right";
                SecureFabLogger.Log("PicoControllerInput", $"Initialized {controllerName} controller");
            }
        }

        private void Update()
        {
            // Re-check controllers if they become disconnected
            if (!activeController.isValid)
            {
                InitializeControllers();
            }

            if (stepManager == null || !stepManager.IsInitialized)
            {
                if (debugLogging && Time.frameCount % 300 == 0)
                {
                    SecureFabLogger.LogWarning("PicoControllerInput", "StepManager not initialized - waiting...");
                }
                return;
            }

            HandleControllerInput();
        }

        private void UpdateActiveController()
        {
            activeController = controllerHand == PXR_Input.Controller.LeftController
                ? leftController
                : rightController;
        }

        private void HandleControllerInput()
        {
            if (!activeController.isValid)
                return;

            // Get current button states
            bool primaryButton, secondaryButton, gripButton, triggerButton;

            // Use Unity XR Input System
            activeController.TryGetFeatureValue(CommonUsages.primaryButton, out primaryButton);
            activeController.TryGetFeatureValue(CommonUsages.secondaryButton, out secondaryButton);
            activeController.TryGetFeatureValue(CommonUsages.gripButton, out gripButton);
            activeController.TryGetFeatureValue(CommonUsages.triggerButton, out triggerButton);

            // PRIMARY BUTTON - Next Step (GetKeyDown pattern)
            if (primaryButton && !lastPrimaryButton)
            {
                OnNextStepPressed();
            }

            // SECONDARY BUTTON - Previous Step
            if (secondaryButton && !lastSecondaryButton)
            {
                OnPreviousStepPressed();
            }

            // GRIP BUTTON - Simulate Correct Configuration
            if (gripButton && !lastGripButton)
            {
                OnSimulateCorrectConfig();
            }

            // TRIGGER - Reset to First Step
            if (triggerButton && !lastTriggerButton)
            {
                OnResetToFirstStep();
            }

            // THUMBSTICK - Jump to specific steps (with cooldown)
            if (Time.time - lastThumbstickTime > thumbstickCooldown)
            {
                Vector2 thumbstick;
                if (activeController.TryGetFeatureValue(CommonUsages.primary2DAxis, out thumbstick))
                {
                    if (thumbstick.y > thumbstickThreshold)
                    {
                        OnJumpToStepIncrement();
                        lastThumbstickTime = Time.time;
                    }
                    else if (thumbstick.y < -thumbstickThreshold)
                    {
                        OnJumpToStepDecrement();
                        lastThumbstickTime = Time.time;
                    }
                }
            }

            // Update last button states
            lastPrimaryButton = primaryButton;
            lastSecondaryButton = secondaryButton;
            lastGripButton = gripButton;
            lastTriggerButton = triggerButton;
        }

        #region Button Action Handlers

        private void OnNextStepPressed()
        {
            SecureFabLogger.Log("PicoControllerInput", "Primary Button: Next Step");
            stepManager.GoToNextStep();
            TriggerHaptic();
        }

        private void OnPreviousStepPressed()
        {
            SecureFabLogger.Log("PicoControllerInput", "Secondary Button: Previous Step");
            stepManager.GoToPreviousStep();
            TriggerHaptic();
        }

        private void OnSimulateCorrectConfig()
        {
            SecureFabLogger.Log("PicoControllerInput", "Grip: Simulating correct configuration");
            
            if (stepManager.CurrentStep != null)
            {
                var correctConfig = stepManager.CurrentStep.expected_config;
                
                SecureFabLogger.Log("PicoControllerInput", "Expected Config:");
                SecureFabLogger.LogConfig("PicoControllerInput", correctConfig);
                
                // Validate the configuration (triggers auto-progress if enabled)
                stepManager.ValidateConfiguration(correctConfig);
            }
            
            TriggerHaptic();
        }

        private void OnResetToFirstStep()
        {
            SecureFabLogger.Log("PicoControllerInput", "Trigger: Reset to first step");
            stepManager.ResetToFirstStep();
            jumpToStepIndex = 0;
            TriggerHaptic(100); // Longer haptic for reset
        }

        private void OnJumpToStepIncrement()
        {
            jumpToStepIndex = (jumpToStepIndex + 1) % stepManager.TotalSteps;
            SecureFabLogger.Log("PicoControllerInput", $"Thumbstick UP: Jump to Step {jumpToStepIndex + 1}");
            stepManager.SetStepById(jumpToStepIndex);
            TriggerHaptic(30);
        }

        private void OnJumpToStepDecrement()
        {
            jumpToStepIndex = (jumpToStepIndex - 1 + stepManager.TotalSteps) % stepManager.TotalSteps;
            SecureFabLogger.Log("PicoControllerInput", $"Thumbstick DOWN: Jump to Step {jumpToStepIndex + 1}");
            stepManager.SetStepById(jumpToStepIndex);
            TriggerHaptic(30);
        }

        #endregion

        #region Haptic Feedback

        private void TriggerHaptic(int durationMs = -1)
        {
            if (!enableHaptics) return;

            int duration = durationMs > 0 ? durationMs : hapticDurationMs;
            
            // Use PICO SDK vibration API
            PXR_Input.SendHapticImpulse(
                PXR_Input.VibrateType.BothController,
                hapticStrength,
                duration,
                (int)controllerHand
            );
        }

        #endregion

        #region Debug UI

        private void OnGUI()
        {
            if (!debugLogging) return;

            GUIStyle style = new GUIStyle();
            style.fontSize = 16;
            style.normal.textColor = Color.cyan;
            style.padding = new RectOffset(10, 10, 10, 10);

            string controllerName = controllerHand == PXR_Input.Controller.LeftController ? "LEFT" : "RIGHT";
            string primaryBtn = controllerHand == PXR_Input.Controller.LeftController ? "X" : "A";
            string secondaryBtn = controllerHand == PXR_Input.Controller.LeftController ? "Y" : "B";
            
            string info = $"PICO CONTROLLER INPUT ({controllerName})\n";
            info += $"Valid: {activeController.isValid}\n\n";
            info += $"{primaryBtn}: Next Step\n";
            info += $"{secondaryBtn}: Previous Step\n";
            info += "GRIP: Simulate Correct Config\n";
            info += "TRIGGER: Reset to Step 1\n";
            info += "THUMBSTICK ↑↓: Jump to Step\n\n";

            if (stepManager != null && stepManager.IsInitialized)
            {
                info += $"Current: Step {stepManager.CurrentStepIndex + 1}/{stepManager.TotalSteps}\n";
                info += $"Jump Target: Step {jumpToStepIndex + 1}";
            }

            GUI.Label(new Rect(10, Screen.height - 220, 400, 220), info, style);
        }

        #endregion
    }
}