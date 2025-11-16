// Assets/Scripts/Custom/SecureFabLogger.cs (enhance existing)
using UnityEngine;
using SecureFab.Training;

public static class SecureFabLogger
{
    private const string TAG_PREFIX = "[SecureFab]";
    private static bool verboseLogging = true;
    
    public static void Log(string component, string message)
    {
        Debug.Log($"{TAG_PREFIX}[{component}] {message}");
    }
    
    public static void LogWarning(string component, string message)
    {
        Debug.LogWarning($"{TAG_PREFIX}[{component}] {message}");
    }
    
    public static void LogError(string component, string message)
    {
        Debug.LogError($"{TAG_PREFIX}[{component}] {message}");
    }
    
    public static void LogVerbose(string component, string message)
    {
        if (verboseLogging)
        {
            Debug.Log($"{TAG_PREFIX}[VERBOSE][{component}] {message}");
        }
    }
    
    public static void LogConfig(string component, ExpectedConfig config)
    {
        Log(component, $"Config: L={config.left ?? "empty"}, R={config.right ?? "empty"}, T={config.top ?? "empty"}, B={config.bottom ?? "empty"}");
    }
    
    public static void LogDetection(string component, string objectName, string zone, float confidence)
    {
        Log(component, $"DETECTED: {objectName} in {zone} (conf: {confidence:F2})");
    }
}