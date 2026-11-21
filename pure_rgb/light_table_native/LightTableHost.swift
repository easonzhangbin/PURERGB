import AppKit
import Foundation

// PURE RGB uses the same Chinese vocabulary in its Qt and native controls.
enum L10n {
    static let locale = Locale(identifier: "zh_CN")
    static func surface(_ text: String) -> String { text }
}

@MainActor enum StudioStyle {
    static let amber = NSColor(srgbRed: 1, green: 0.49, blue: 0.29, alpha: 1)
    static let charcoal = NSColor(white: 0.08, alpha: 1)
    static let secondaryText = NSColor.secondaryLabelColor

    static func embed(_ view: NSView, in host: NSView, horizontal: CGFloat, vertical: CGFloat) {
        view.translatesAutoresizingMaskIntoConstraints = false
        host.addSubview(view)
        NSLayoutConstraint.activate([
            view.leadingAnchor.constraint(equalTo: host.leadingAnchor, constant: horizontal),
            view.trailingAnchor.constraint(equalTo: host.trailingAnchor, constant: -horizontal),
            view.topAnchor.constraint(equalTo: host.topAnchor, constant: vertical),
            view.bottomAnchor.constraint(equalTo: host.bottomAnchor, constant: -vertical)
        ])
    }

    static func installBackdrop(in window: NSWindow, root: NSView) {
        window.isOpaque = false
        window.backgroundColor = .clear
        embed(PUREUltraThinBridge.makeThickView(), in: root, horizontal: 0, vertical: 0)
    }

    static func wordmark() -> NSView {
        let helperDirectory = URL(fileURLWithPath: CommandLine.arguments[0])
            .resolvingSymlinksInPath().deletingLastPathComponent()
        let logoURL: URL
        if helperDirectory.lastPathComponent == "Helpers" {
            logoURL = helperDirectory.deletingLastPathComponent()
                .appendingPathComponent("Resources/Source/assets/branding/pure-rgb-header.png")
        } else {
            logoURL = helperDirectory.appendingPathComponent("pure-rgb-header.png")
        }
        guard let image = NSImage(contentsOf: logoURL) else {
            return NSTextField(labelWithString: "PURE RGB")
        }
        image.isTemplate = false
        let view = NSImageView(image: image)
        view.imageScaling = .scaleProportionallyUpOrDown
        view.setAccessibilityLabel("PURE RGB Logo")
        NSLayoutConstraint.activate([
            view.widthAnchor.constraint(equalToConstant: 128),
            view.heightAnchor.constraint(equalToConstant: 32)
        ])
        return view
    }

    static func panel(_ view: NSView, horizontal: CGFloat, vertical: CGFloat) -> NSView {
        let host = NSView()
        embed(view, in: host, horizontal: horizontal, vertical: vertical)
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            let glass = NSGlassEffectView()
            glass.style = .clear
            glass.cornerRadius = 18
            glass.contentView = host
            // The content view's intrinsic content size is insufficient for stack layouts.
            NSLayoutConstraint.activate([
                glass.widthAnchor.constraint(equalTo: host.widthAnchor),
                glass.heightAnchor.constraint(equalTo: host.heightAnchor)
            ])
            return glass
        }
        #endif
        let panel = NSView()
        panel.wantsLayer = true
        panel.layer?.cornerRadius = 18
        panel.layer?.masksToBounds = true
        embed(PUREUltraThinBridge.makeView(), in: panel, horizontal: 0, vertical: 0)
        embed(host, in: panel, horizontal: 0, vertical: 0)
        return panel
    }

    static func actionControl(_ button: NSButton, horizontal: CGFloat = 12) -> NSView {
        button.bezelStyle = .rounded
        button.controlSize = .regular
        return button
    }

    static func backControl(_ button: NSButton) -> NSView {
        button.image = NSImage(systemSymbolName: "chevron.left", accessibilityDescription: nil)
        button.imagePosition = .imageLeading
        return actionControl(button)
    }
}
