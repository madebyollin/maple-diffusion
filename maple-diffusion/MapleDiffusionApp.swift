import SwiftUI

@main
struct MapleDiffusionApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(
                    minWidth: 192,
                    minHeight: 192
                )
                .navigationTitle("🍁 Maple Diffusion")
        }

        #if os(macOS)
        Settings {
            PreferencesView()
        }
        #endif
    }
}
