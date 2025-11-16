using System;
using UnityEngine;
using UnityEngine.Events;

namespace SecureFab.Training
{
    /// <summary>
    /// Manages the training procedure steps, loads from JSON, and tracks progress.
    /// Attach to a GameObject in your scene and configure in the Inspector.
    /// </summary>
    public class StepManager : MonoBehaviour
    {
        #region Inspector Fields

        [Header("JSON Settings")]
        [Tooltip("Name of the JSON file in Resources, without extension")]
        public string stepsJsonResourceName = "steps";

        [Header("UI References (Optional)")]
        [Tooltip("TextMeshPro component to display step title")]
        public TMPro.TextMeshProUGUI stepTitleText;

        [Tooltip("TextMeshPro component to display step body/instructions")]
        public TMPro.TextMeshProUGUI stepBodyText;

        [Header("Auto-Progress Settings")]
        [Tooltip("Allow automatic step progression when config matches")]
        public bool enableAutoProgress = false;

        [Tooltip("Delay in seconds before auto-advancing to next step")]
        [Range(0f, 5f)]
        public float autoProgressDelay = 1.0f;

        [Header("Events")]
        [Tooltip("Invoked when the current step changes")]
        public UnityEvent<Step> onStepChanged;

        [Tooltip("Invoked when the procedure is completed")]
        public UnityEvent onProcedureComplete;

        [Tooltip("Invoked when step configuration is validated")]
        public UnityEvent<bool> onConfigurationValidated;

        #endregion

        #region Private Fields

        private StepListWrapper _stepList;
        private int _currentStepIndex = 0;
        private bool _isInitialized = false;
        private Step _cachedCurrentStep;
        private float _autoProgressTimer = 0f;
        private bool _waitingForAutoProgress = false;

        #endregion

        #region Public Properties

        /// <summary>
        /// Get the current step. Cached for performance.
        /// </summary>
        public Step CurrentStep
        {
            get
            {
                if (_cachedCurrentStep == null || _cachedCurrentStep.id != _currentStepIndex)
                {
                    UpdateCachedCurrentStep();
                }
                return _cachedCurrentStep;
            }
        }

        /// <summary>
        /// Get the current step index (0-based).
        /// </summary>
        public int CurrentStepIndex => _currentStepIndex;

        /// <summary>
        /// Get the total number of steps.
        /// </summary>
        public int TotalSteps => _stepList?.Count ?? 0;

        /// <summary>
        /// Check if there is a next step available.
        /// </summary>
        public bool HasNextStep => _currentStepIndex < TotalSteps - 1;

        /// <summary>
        /// Check if there is a previous step available.
        /// </summary>
        public bool HasPreviousStep => _currentStepIndex > 0;

        /// <summary>
        /// Check if the procedure is complete (on last step).
        /// </summary>
        public bool IsComplete => _isInitialized && _currentStepIndex >= TotalSteps - 1;

        /// <summary>
        /// Get the progress as a percentage (0-100).
        /// </summary>
        public float ProgressPercentage => TotalSteps > 0 ? ((_currentStepIndex + 1f) / TotalSteps) * 100f : 0f;

        /// <summary>
        /// Check if the manager is properly initialized.
        /// </summary>
        public bool IsInitialized => _isInitialized;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            LoadStepsFromResources();
        }

        private void Start()
        {
            if (_isInitialized)
            {
                ApplyCurrentStepToUI();
            }
        }

        private void Update()
        {
            if (_waitingForAutoProgress && enableAutoProgress)
            {
                _autoProgressTimer += Time.deltaTime;
                if (_autoProgressTimer >= autoProgressDelay)
                {
                    _waitingForAutoProgress = false;
                    _autoProgressTimer = 0f;
                    GoToNextStep();
                }
            }
        }

        #endregion

        #region Step Loading

        /// <summary>
        /// Load steps from Resources folder.
        /// </summary>
        private void LoadStepsFromResources()
        {
            try
            {
                TextAsset jsonAsset = Resources.Load<TextAsset>(stepsJsonResourceName);
                if (jsonAsset == null)
                {
                    Debug.LogError($"[StepManager] Could not load Resources/{stepsJsonResourceName}.json");
                    return;
                }

                _stepList = JsonUtility.FromJson<StepListWrapper>(jsonAsset.text);

                if (_stepList == null)
                {
                    Debug.LogError("[StepManager] Failed to deserialize JSON. Check format.");
                    return;
                }

                if (!_stepList.IsValid())
                {
                    Debug.LogError("[StepManager] Step list validation failed. Check JSON data.");
                    return;
                }

                _currentStepIndex = Mathf.Clamp(_currentStepIndex, 0, TotalSteps - 1);
                UpdateCachedCurrentStep();
                _isInitialized = true;

                Debug.Log($"[StepManager] Successfully loaded {TotalSteps} steps from {stepsJsonResourceName}.json");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[StepManager] Exception while loading steps: {ex.Message}");
            }
        }

        #endregion

        #region Step Navigation

        /// <summary>
        /// Advance to the next step.
        /// </summary>
        public void GoToNextStep()
        {
            if (!_isInitialized)
            {
                Debug.LogWarning("[StepManager] Cannot navigate - not initialized.");
                return;
            }

            if (HasNextStep)
            {
                _currentStepIndex++;
                OnStepIndexChanged();
            }
            else if (IsComplete)
            {
                Debug.Log("[StepManager] Procedure complete!");
                onProcedureComplete?.Invoke();
            }
            else
            {
                Debug.LogWarning("[StepManager] Already on the last step.");
            }
        }

        /// <summary>
        /// Go back to the previous step.
        /// </summary>
        public void GoToPreviousStep()
        {
            if (!_isInitialized)
            {
                Debug.LogWarning("[StepManager] Cannot navigate - not initialized.");
                return;
            }

            if (HasPreviousStep)
            {
                _currentStepIndex--;
                OnStepIndexChanged();
            }
            else
            {
                Debug.LogWarning("[StepManager] Already on the first step.");
            }
        }

        /// <summary>
        /// Jump to a specific step by ID.
        /// </summary>
        public void SetStepById(int id)
        {
            if (!_isInitialized)
            {
                Debug.LogWarning("[StepManager] Cannot navigate - not initialized.");
                return;
            }

            Step targetStep = _stepList.FindStepById(id);
            if (targetStep != null)
            {
                _currentStepIndex = id;
                OnStepIndexChanged();
            }
            else
            {
                Debug.LogWarning($"[StepManager] Step with ID {id} not found.");
            }
        }

        /// <summary>
        /// Jump to a specific step by index.
        /// </summary>
        public void SetStepByIndex(int index)
        {
            if (!_isInitialized)
            {
                Debug.LogWarning("[StepManager] Cannot navigate - not initialized.");
                return;
            }

            if (index >= 0 && index < TotalSteps)
            {
                _currentStepIndex = index;
                OnStepIndexChanged();
            }
            else
            {
                Debug.LogWarning($"[StepManager] Index {index} out of range [0, {TotalSteps - 1}].");
            }
        }

        /// <summary>
        /// Reset to the first step.
        /// </summary>
        public void ResetToFirstStep()
        {
            if (!_isInitialized) return;

            _currentStepIndex = 0;
            OnStepIndexChanged();
            Debug.Log("[StepManager] Reset to first step.");
        }

        #endregion

        #region Configuration Validation

        /// <summary>
        /// Validate a detected configuration against the current step's expected config.
        /// </summary>
        /// <param name="detectedConfig">The configuration detected by SecureMR</param>
        /// <returns>True if the configuration matches expectations</returns>
// In StepManager.cs
public bool ValidateConfiguration(ExpectedConfig detectedConfig)
{
    if (!_isInitialized || CurrentStep == null)
    {
        Debug.LogWarning("[StepManager] Cannot validate - not initialized or no current step.");
        return false;
    }
    bool matches = CurrentStep.expected_config.Matches(detectedConfig);

    // Enhanced logging
    SecureFabLogger.Log("StepManager", $"=== CONFIG VALIDATION ===");
    SecureFabLogger.Log("StepManager", $"Step: {CurrentStep.id} - {CurrentStep.title}");
    SecureFabLogger.LogConfig("StepManager-Expected", CurrentStep.expected_config);
    SecureFabLogger.LogConfig("StepManager-Detected", detectedConfig);
    SecureFabLogger.Log("StepManager", $"Result: {(matches ? "✓ PASS" : "✗ FAIL")}");
    SecureFabLogger.Log("StepManager", "=====================");

    onConfigurationValidated?.Invoke(matches);

    if (matches && enableAutoProgress && HasNextStep && !_waitingForAutoProgress)
    {
        _waitingForAutoProgress = true;
        _autoProgressTimer = 0f;
        SecureFabLogger.Log("StepManager", $"Auto-progress will activate in {autoProgressDelay}s");
    }

    return matches;
}

        /// <summary>
        /// Cancel pending auto-progress.
        /// </summary>
        public void CancelAutoProgress()
        {
            _waitingForAutoProgress = false;
            _autoProgressTimer = 0f;
        }

        #endregion

        #region UI Updates

        private void OnStepIndexChanged()
        {
            UpdateCachedCurrentStep();
            CancelAutoProgress();
            ApplyCurrentStepToUI();
        }

        private void UpdateCachedCurrentStep()
        {
            if (_stepList != null && _currentStepIndex >= 0 && _currentStepIndex < TotalSteps)
            {
                _cachedCurrentStep = _stepList.steps[_currentStepIndex];
            }
            else
            {
                _cachedCurrentStep = null;
            }
        }

        private void ApplyCurrentStepToUI()
        {
            Step step = CurrentStep;
            if (step == null)
            {
                Debug.LogWarning("[StepManager] No current step to display.");
                return;
            }

            // Update UI text components if assigned
            if (stepTitleText != null)
            {
                stepTitleText.text = step.title;
            }

            if (stepBodyText != null)
            {
                stepBodyText.text = step.body;
            }

            // Invoke event
            onStepChanged?.Invoke(step);

            // Log step change
            Debug.Log($"[StepManager] Step {_currentStepIndex + 1}/{TotalSteps}: {step.title}");
        }

        #endregion

        #region Public Utility Methods

        /// <summary>
        /// Get a step by ID. Returns null if not found.
        /// </summary>
        public Step GetStepById(int id)
        {
            return _stepList?.FindStepById(id);
        }

        /// <summary>
        /// Get a step by index. Returns null if out of range.
        /// </summary>
        public Step GetStepByIndex(int index)
        {
            if (_stepList?.steps != null && index >= 0 && index < _stepList.steps.Length)
            {
                return _stepList.steps[index];
            }
            return null;
        }

        /// <summary>
        /// Get a formatted progress string (e.g., "Step 3 of 5").
        /// </summary>
        public string GetProgressString()
        {
            return $"Step {_currentStepIndex + 1} of {TotalSteps}";
        }

        #endregion

        #region Debug Helper

#if UNITY_EDITOR
        [ContextMenu("Log All Steps")]
        private void LogAllSteps()
        {
            if (!_isInitialized)
            {
                Debug.Log("[StepManager] Not initialized. Cannot log steps.");
                return;
            }

            Debug.Log($"[StepManager] === All Steps ({TotalSteps}) ===");
            for (int i = 0; i < TotalSteps; i++)
            {
                var step = _stepList.steps[i];
                Debug.Log($"  {step.GetSummary()}");
            }
        }

        [ContextMenu("Next Step")]
        private void DebugNextStep() => GoToNextStep();

        [ContextMenu("Previous Step")]
        private void DebugPreviousStep() => GoToPreviousStep();

        [ContextMenu("Reset to First")]
        private void DebugReset() => ResetToFirstStep();
#endif

        #endregion
    }
}
