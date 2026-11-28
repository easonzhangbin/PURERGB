import AppKit
import Foundation
import QuartzCore

// PURE RGB uses the same Chinese vocabulary in its Qt and native controls.
enum L10n {
    static let locale = Locale(identifier: "zh_CN")
    static func surface(_ text: String) -> String { text }
}

@MainActor enum StudioStyle {
    static let amber = NSColor(srgbRed: 1, green: 0.49, blue: 0.29, alpha: 1)
    static let charcoal = NSColor(white: 0.08, alpha: 1)
    static let secondaryText = NSColor.secondaryLabelColor
    private static let obsidianChassis = NSColor(srgbRed: 11 / 255.0, green: 11 / 255.0,
                                                 blue: 12 / 255.0, alpha: 1)

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

    static func installBackdrop(in window: NSWindow, root: NSView, obsidian: Bool) {
        if obsidian {
            window.isOpaque = true
            window.backgroundColor = obsidianChassis
            return
        }
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

    static func panel(_ view: NSView, horizontal: CGFloat, vertical: CGFloat,
                      obsidian: Bool, accented: Bool = false) -> NSView {
        if obsidian {
            let panel = ObsidianLightTableBar(accented: accented)
            embed(view, in: panel, horizontal: horizontal, vertical: vertical)
            return panel
        }
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

private final class ObsidianLightTableBar: NSView {
    private let accent: CAGradientLayer?

    init(accented: Bool) {
        accent = accented ? CAGradientLayer() : nil
        super.init(frame: .zero)
        wantsLayer = true
        layer?.backgroundColor = CGColor(srgbRed: 20 / 255.0, green: 21 / 255.0,
                                         blue: 23 / 255.0, alpha: 1)
        layer?.borderColor = CGColor(srgbRed: 52 / 255.0, green: 54 / 255.0,
                                     blue: 60 / 255.0, alpha: 1)
        layer?.borderWidth = 1
        layer?.cornerRadius = 12
        if let accent {
            accent.startPoint = CGPoint(x: 0, y: 0.5)
            accent.endPoint = CGPoint(x: 1, y: 0.5)
            accent.locations = [0, 0.48, 0.82, 1]
            accent.colors = [
                CGColor(srgbRed: 0.44, green: 0.49, blue: 0.86, alpha: 0),
                CGColor(srgbRed: 0.57, green: 0.55, blue: 0.88, alpha: 0.38),
                CGColor(srgbRed: 0.94, green: 0.63, blue: 0.59, alpha: 0.66),
                CGColor(srgbRed: 0.94, green: 0.63, blue: 0.59, alpha: 0),
            ]
            layer?.addSublayer(accent)
        }
    }

    required init?(coder: NSCoder) { nil }

    override func layout() {
        super.layout()
        guard let accent else { return }
        let width = min(172, bounds.width * 0.24)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        accent.frame = CGRect(x: bounds.maxX - width - 24, y: bounds.maxY - 1.6,
                              width: width, height: 1.1)
        CATransaction.commit()
    }
}
