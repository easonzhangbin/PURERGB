import AppKit
import SwiftUI

@available(macOS 12.0, *)
private struct GlassSurface: View {
    let material: Material

    var body: some View {
        Rectangle().fill(material).ignoresSafeArea().allowsHitTesting(false)
    }
}

@available(macOS 12.0, *)
private final class PassiveMaterialView: NSHostingView<GlassSurface> {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
    override var isOpaque: Bool { false }
}

@available(macOS 12.0, *)
@objc(PUREUltraThinBridge)
final class PUREUltraThinBridge: NSObject {
    @objc @MainActor static func makeView() -> NSView {
        return makeSurface(.ultraThinMaterial)
    }

    @objc @MainActor static func makeThickView() -> NSView {
        return makeSurface(.thickMaterial)
    }

    @MainActor private static func makeSurface(_ material: Material) -> NSView {
        let view = PassiveMaterialView(rootView: GlassSurface(material: material))
        view.autoresizingMask = [.width, .height]
        return view
    }
}
