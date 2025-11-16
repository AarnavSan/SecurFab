using UnityEngine;
using UnityEngine.UI;
using TMPro;
using SecureFab.Training;

namespace PicoXR.SecureMR.Demo
{
    /// <summary>
    /// Visual progress indicator for the SecureFab training system.
    /// Shows step progress, current configuration status, and validation feedback.
    /// </summary>
    public class TrainingProgressUI : MonoBehaviour
    {
        [Header("UI Elements")]
        public Image progressBar;
        public TextMeshProUGUI progressText;
        public TextMeshProUGUI statusText;
        public Image validationIndicator;

        [Header("Colors")]
        public Color correctColor = Color.green;
        public Color incorrectColor = Color.red;
        public Color neutralColor = Color.yellow;
        public Color progressColor = Color.blue;

        [Header("References")]
        public StepManager stepManager;

        [Header("Animation")]
        public bool animateProgressBar = true;
        public float animationSpeed = 2f;

        private float targetProgress = 0f;
        private float currentProgress = 0f;
        private ValidationState currentState = ValidationState.Neutral;

        private enum ValidationState
        {
            Neutral,
            Correct,
            Incorrect
        }

        private void Start()
        {
            if (stepManager != null)
            {
                stepManager.onStepChanged.AddListener(OnStepChanged);
                stepManager.onConfigurationValidated.AddListener(OnConfigValidated);
                
                UpdateProgress();
            }
        }

        private void Update()
        {
            if (animateProgressBar)
            {
                // Smooth progress bar animation
                currentProgress = Mathf.Lerp(currentProgress, targetProgress, Time.deltaTime * animationSpeed);
                
                if (progressBar != null)
                {
                    progressBar.fillAmount = currentProgress;
                }
            }
        }

        private void OnStepChanged(Step newStep)
        {
            UpdateProgress();
            SetValidationState(ValidationState.Neutral);
        }

        private void OnConfigValidated(bool isValid)
        {
            SetValidationState(isValid ? ValidationState.Correct : ValidationState.Incorrect);

            // Auto-reset validation indicator after delay
            if (!isValid)
            {
                Invoke(nameof(ResetValidationIndicator), 2f);
            }
        }

        private void UpdateProgress()
        {
            if (stepManager == null) return;

            // Calculate progress
            float progress = stepManager.ProgressPercentage / 100f;
            targetProgress = progress;

            if (!animateProgressBar && progressBar != null)
            {
                progressBar.fillAmount = progress;
            }

            // Update progress text
            if (progressText != null)
            {
                progressText.text = stepManager.GetProgressString();
            }

            // Update status text
            if (statusText != null && stepManager.CurrentStep != null)
            {
                statusText.text = stepManager.CurrentStep.title;
            }

            // Update progress bar color
            if (progressBar != null)
            {
                progressBar.color = progressColor;
            }
        }

        private void SetValidationState(ValidationState state)
        {
            currentState = state;

            if (validationIndicator == null) return;

            switch (state)
            {
                case ValidationState.Correct:
                    validationIndicator.color = correctColor;
                    SecureFabLogger.Log("ProgressUI", "✓ Configuration CORRECT");
                    break;

                case ValidationState.Incorrect:
                    validationIndicator.color = incorrectColor;
                    SecureFabLogger.Log("ProgressUI", "✗ Configuration INCORRECT");
                    break;

                case ValidationState.Neutral:
                    validationIndicator.color = neutralColor;
                    break;
            }
        }

        private void ResetValidationIndicator()
        {
            SetValidationState(ValidationState.Neutral);
        }

        private void OnDestroy()
        {
            if (stepManager != null)
            {
                stepManager.onStepChanged.RemoveListener(OnStepChanged);
                stepManager.onConfigurationValidated.RemoveListener(OnConfigValidated);
            }
        }
    }
}