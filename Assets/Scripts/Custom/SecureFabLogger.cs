// Assets/Scripts/SecureFabLogger.cs
using UnityEngine;
using SecureFab.Training;

public static class SecureFabLogger
{
    private const string TAG_PREFIX = "[SecureFab]";
    
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
    
    public static void LogConfig(string component, ExpectedConfig config)
    {
        Log(component, $"Config: L={config.left ?? "empty"}, R={config.right ?? "empty"}, T={config.top ?? "empty"}, B={config.bottom ?? "empty"}");
    }
}