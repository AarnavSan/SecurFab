using System;
using UnityEngine;

namespace SecureFab.Training
{
    /// <summary>
    /// Represents the expected object configuration for a training step.
    /// Maps to the four zones on the Tool Simulation Board.
    /// </summary>
    [Serializable]
    public class ExpectedConfig
    {
        public string left;
        public string right;
        public string top;
        public string bottom;

        /// <summary>
        /// Check if a specific zone has an expected object.
        /// </summary>
        public bool HasObjectInZone(string zone)
        {
            return GetZoneObject(zone) != null;
        }

        /// <summary>
        /// Get the expected object for a specific zone.
        /// </summary>
        public string GetZoneObject(string zone)
        {
            return zone?.ToLower() switch
            {
                "left" => left,
                "right" => right,
                "top" => top,
                "bottom" => bottom,
                _ => null
            };
        }

        /// <summary>
        /// Check if this configuration matches another configuration.
        /// </summary>
        public bool Matches(ExpectedConfig other)
        {
            if (other == null) return false;

            return StringEquals(left, other.left) &&
                   StringEquals(right, other.right) &&
                   StringEquals(top, other.top) &&
                   StringEquals(bottom, other.bottom);
        }

        /// <summary>
        /// Compare two strings, treating null and empty as equivalent.
        /// </summary>
        private bool StringEquals(string a, string b)
        {
            return string.IsNullOrEmpty(a) ? string.IsNullOrEmpty(b) : a.Equals(b);
        }

        /// <summary>
        /// Count how many zones have objects.
        /// </summary>
        public int GetObjectCount()
        {
            int count = 0;
            if (!string.IsNullOrEmpty(left)) count++;
            if (!string.IsNullOrEmpty(right)) count++;
            if (!string.IsNullOrEmpty(top)) count++;
            if (!string.IsNullOrEmpty(bottom)) count++;
            return count;
        }

        /// <summary>
        /// Check if all zones are empty.
        /// </summary>
        public bool IsEmpty()
        {
            return string.IsNullOrEmpty(left) &&
                   string.IsNullOrEmpty(right) &&
                   string.IsNullOrEmpty(top) &&
                   string.IsNullOrEmpty(bottom);
        }

        public override string ToString()
        {
            return $"[L:{left ?? "empty"}, R:{right ?? "empty"}, T:{top ?? "empty"}, B:{bottom ?? "empty"}]";
        }
    }

    /// <summary>
    /// Represents a single training step in the procedure.
    /// </summary>
    [Serializable]
    public class Step
    {
        public int id;
        public string title;
        public string body;
        public ExpectedConfig expected_config;

        /// <summary>
        /// Check if this is a valid step (has required fields).
        /// </summary>
        public bool IsValid()
        {
            return id >= 0 &&
                   !string.IsNullOrEmpty(title) &&
                   !string.IsNullOrEmpty(body) &&
                   expected_config != null;
        }

        /// <summary>
        /// Get a condensed summary of this step for logging.
        /// </summary>
        public string GetSummary()
        {
            return $"Step {id}: {title} - Config: {expected_config}";
        }
    }

    /// <summary>
    /// Wrapper for deserializing the steps.json file.
    /// JsonUtility requires a wrapper class to deserialize root-level arrays.
    /// </summary>
    [Serializable]
    public class StepListWrapper
    {
        public Step[] steps;

        /// <summary>
        /// Get the number of steps.
        /// </summary>
        public int Count => steps?.Length ?? 0;

        /// <summary>
        /// Check if the step list is valid.
        /// </summary>
        public bool IsValid()
        {
            if (steps == null || steps.Length == 0)
                return false;

            // Verify all steps are valid and IDs are sequential
            for (int i = 0; i < steps.Length; i++)
            {
                if (!steps[i].IsValid() || steps[i].id != i)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Find a step by its ID.
        /// </summary>
        public Step FindStepById(int id)
        {
            if (steps == null) return null;

            // Since IDs should be sequential, try direct indexing first
            if (id >= 0 && id < steps.Length && steps[id].id == id)
                return steps[id];

            // Fallback to linear search
            foreach (var step in steps)
            {
                if (step.id == id)
                    return step;
            }

            return null;
        }
    }
}
