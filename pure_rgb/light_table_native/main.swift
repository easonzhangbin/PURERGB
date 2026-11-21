import AppKit
import Foundation

private enum Mode: String, Decodable { case single, roll }

private struct Frame: Decodable {
    let id: UUID
    let artwork: String
    let name: String
    let status: String
    var selected: Bool

    var item: BatchLightTableItem {
        BatchLightTableItem(id: id, url: URL(fileURLWithPath: artwork), status: status,
                           selected: selected, displayName: name)
    }
}

private struct Manifest: Decodable {
    let mode: Mode
    let canExport: Bool
    let current: UUID?
    let displayID: CGDirectDisplayID?
    let items: [Frame]
}

private struct Reply: Encodable {
    let action: String
    var id: UUID?
    var selected: Bool?

    func send() {
        if var bytes = try? JSONEncoder().encode(self) {
            bytes.append(0x0a)
            FileHandle.standardOutput.write(bytes)
        }
    }
}

@MainActor private final class LightTableSession: NSObject, NSApplicationDelegate {
    let manifest: Manifest
    let controller: BatchLightTableController
    var frames: [Frame]
    private var ending = false

    init(_ manifest: Manifest) throws {
        guard !manifest.items.isEmpty, Set(manifest.items.map(\.id)).count == manifest.items.count,
              manifest.items.allSatisfy({ ($0.artwork as NSString).isAbsolutePath }) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.manifest = manifest
        if manifest.mode == .single {
            let active = manifest.current ?? manifest.items[0].id
            guard let frame = manifest.items.first(where: { $0.id == active }) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            frames = [frame]
        } else {
            frames = manifest.items
        }
        controller = BatchLightTableController(viewingRepair: manifest.mode == .single)
        super.init()
        controller.onNext = { [weak self] in
            guard let self, self.manifest.canExport else { return }
            self.finish("export")
        }
        controller.onReturn = { [weak self] in self?.finish("return") }
        controller.onSelect = { [weak self] id in self?.toggle(id) }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        refresh()
        controller.present(on: NSScreen.screens.first { $0.lightTableDisplayID == manifest.displayID })
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationDidChangeScreenParameters(_ notification: Notification) {
        controller.updateDisplay()
    }

    private func refresh() {
        let status = manifest.mode == .single ? [frames[0].name, frames[0].status].joined(separator: " · ") : ""
        controller.update(items: frames.map(\.item), status: status, busy: false, canAdvance: manifest.canExport)
    }

    private func toggle(_ id: UUID) {
        guard let index = frames.firstIndex(where: { $0.id == id }) else { return }
        frames[index].selected.toggle()
        refresh()
        Reply(action: "select", id: id, selected: frames[index].selected).send()
    }

    private func finish(_ action: String) {
        guard !ending else { return }
        ending = true
        Reply(action: action).send()
        NSApp.terminate(nil)
    }
}

do {
    guard CommandLine.arguments.count == 2 else { throw CocoaError(.fileReadInvalidFileName) }
    let bytes = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    let manifest = try JSONDecoder().decode(Manifest.self, from: bytes)
    try MainActor.assumeIsolated {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let session = try LightTableSession(manifest)
        app.delegate = session
        withExtendedLifetime(session) { app.run() }
    }
} catch {
    FileHandle.standardError.write(Data("无法打开观片台：\(error.localizedDescription)\n".utf8))
    exit(1)
}
