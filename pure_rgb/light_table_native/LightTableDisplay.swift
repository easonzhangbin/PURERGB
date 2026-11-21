import AppKit
import CoreGraphics

extension NSScreen {
    var lightTableDisplayID: CGDirectDisplayID? {
        (deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value
    }

    @MainActor var isAppleXDR: Bool {
        get async {
            guard let id = lightTableDisplayID else { return false }
            if localizedName.localizedCaseInsensitiveContains("XDR"), CGDisplayVendorNumber(id) == 0x0610 {
                return true
            }
            guard CGDisplayIsBuiltin(id) != 0 else { return false }
            return await internalPanelIsXDR.value
        }
    }
}

// AppKit can call an XDR panel "Built-in Retina Display". EDR headroom is a
// property of the current display mode and is also available on non-XDR Macs.
// Consult the system's hardware description once, off the UI thread, only
// when the viewer actually uses an internal display with a generic name.
private let internalPanelIsXDR = Task.detached(priority: .userInitiated) {
    let profiler = Process()
    let output = Pipe()
    profiler.executableURL = URL(fileURLWithPath: "/usr/sbin/system_profiler")
    profiler.arguments = ["SPDisplaysDataType", "-json", "-timeout", "4"]
    profiler.standardOutput = output
    profiler.standardError = FileHandle.nullDevice
    do { try profiler.run() } catch { return false }
    let data = output.fileHandleForReading.readDataToEndOfFile()
    profiler.waitUntilExit()
    guard profiler.terminationStatus == 0,
          let report = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let adapters = report["SPDisplaysDataType"] as? [[String: Any]] else { return false }
    return adapters.contains { gpu in
        let panels = gpu["spdisplays_ndrvs"] as? [[String: Any]] ?? []
        return panels.contains { panel in
            guard panel["spdisplays_connection_type"] as? String == "spdisplays_internal",
                  let type = panel["spdisplays_display_type"] as? String else { return false }
            return type.localizedCaseInsensitiveContains("xdr")
        }
    }
}
