import AppKit
import SwiftUI

// The desktop wallpaper is supplied by the AppKit behind-window effect below
// this view. SwiftUI's material itself only blurs content within the app.
@available(macOS 12.0, *)
private struct UltraThinSurface: View {
    var body: some View {
        Rectangle().fill(.ultraThinMaterial).ignoresSafeArea().allowsHitTesting(false)
    }
}

@available(macOS 12.0, *)
private final class PassiveMaterialView: NSHostingView<UltraThinSurface> {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
    override var isOpaque: Bool { false }
}

@available(macOS 12.0, *)
@objc(PUREUltraThinBridge)
final class PUREUltraThinBridge: NSObject {
    @objc @MainActor static func makeView() -> NSView {
        let view = PassiveMaterialView(rootView: UltraThinSurface())
        view.autoresizingMask = [.width, .height]
        return view
    }
}
