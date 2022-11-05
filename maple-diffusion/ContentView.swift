import SwiftUI

#if os(macOS)
import Foundation

extension NSImage {
    var pngData: Data? {
        guard let tiffRepresentation = tiffRepresentation, let bitmapImage = NSBitmapImageRep(data: tiffRepresentation) else { return nil }
        return bitmapImage.representation(using: .png, properties: [:])
    }
    func pngWrite(
        to url: URL,
        options: Data.WritingOptions = .atomic
    ) {
        do {
            try pngData?.write(to: url, options: options)
        } catch {
            print(error)
        }
    }
}
#else
import UIKit
#endif

// MARK: - ContentView

struct ContentView: View {
    #if os(iOS)
    let mapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: true)
    #else
    let mapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: false)
    #endif
    let dispatchQueue = DispatchQueue(label: "Generation")
    @State var image: Image?
    @State var cgImage: CGImage?
    @State var prompt: String = ""
    @State var negativePrompt: String = ""
    @State var running: Bool = false
    @State var progressProp: Float = 1
    @State var progressStage: String = "Ready"
    
    @AppStorage("nsfw") var allowNSFWContent: Bool = false
    @AppStorage("nai") var useNovelAiPrompts: Bool = true
    @AppStorage("steps") var steps: Double = 28
    @AppStorage("guidance") var guidanceScale: Double = 11.0
    
    let novelAIPrompt: String = "masterpiece, best quality, "
    let novelAIAntiprompt: String = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, "

    #if os(macOS)
    func showSavePanel() {
        let savePanel: NSSavePanel = .init()
        
        savePanel.allowedContentTypes = [.png]
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.nameFieldStringValue = String(UUID().uuidString.prefix(8))
        savePanel.title = "Save generated artwork"
        savePanel.message = "Choose a file name to store image"
        
        let response: NSApplication.ModalResponse = savePanel.runModal()
        let url: URL? = response == .OK ? savePanel.url : nil
        
        if let url = url {
            let nsImage: NSImage = .init(
                cgImage: cgImage!,
                size: NSSize(width: 768, height: 768)
            )
            
            nsImage.pngWrite(to: url)
        }
    }
    #else
    func showSavePanel() {
        let uiImage: UIImage = .init(cgImage: cgImage!)
        UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
    }
    #endif
    
    var body: some View {
        VStack {
            if image == nil {
                Rectangle()
                    .fill(.gray)
                    .aspectRatio(1.0, contentMode: .fit)
                    .frame(
                        idealWidth: mapleDiffusion.width as? CGFloat,
                        idealHeight: mapleDiffusion.height as? CGFloat
                    )
            } else {
                image!
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(
                        idealWidth: mapleDiffusion.width as? CGFloat,
                        idealHeight: mapleDiffusion.height as? CGFloat
                    )
                #if os(macOS)
                    .onTapGesture {
                        showSavePanel()
                    }
                #endif
            }
            HStack {
                Text("Prompt").bold()
                TextField("What you want", text: $prompt)
            }
            
            HStack {
                Text("Negative Prompt").bold()
                TextField("What you don't want", text: $negativePrompt)
            }
            
            HStack {
                HStack {
                    Text("Scale").bold()
                    Text(String(format: "%.1f", guidanceScale)).foregroundColor(.secondary)
                }
                .frame(
                    width: 96,
                    alignment: .leading
                )
                
                Slider(value: $guidanceScale, in: 1...20)
            }
            
            HStack {
                HStack {
                    Text("Steps").bold()
                    Text("\(Int(steps))").foregroundColor(.secondary)
                }
                .frame(
                    width: 96,
                    alignment: .leading
                )
                
                Slider(value: $steps, in: 5...150)
            }
            ProgressView(progressStage, value: progressProp, total: 1)
                .opacity(running ? 1 : 0)
                .foregroundColor(.secondary)
            
            Spacer(minLength: 8)
            
            Button(action: generate) {
                Text("Generate Image")
                    .frame(
                        minWidth: 100,
                        maxWidth: .infinity,
                        minHeight: 64,
                        alignment: .center
                    )
                    .background(running ? .gray : .blue)
                    .foregroundColor(.white)
                    .font(Font.title)
                    .cornerRadius(32)
            }
            .buttonStyle(.borderless)
            .disabled(running)
        }
        .padding(16)
        .onAppear {
            loadModels()
        }
    }

    func loadModels() {
        dispatchQueue.async {
            running = true
            mapleDiffusion.initModels { p, s in
                progressProp = p
                progressStage = s
            }
            running = false
        }
    }
    
    func generate() {
        dispatchQueue.async {
            running = true
            progressStage = ""
            progressProp = 0
            mapleDiffusion.generate(
                prompt: "\(useNovelAiPrompts ? novelAIPrompt : "")" + prompt,
                negativePrompt: "\(useNovelAiPrompts ? novelAIAntiprompt : "")" + "\(allowNSFWContent ? "" : "nsfw, ")" + negativePrompt,
                seed: Int.random(in: 1 ..< Int.max),
                steps: Int(steps),
                guidanceScale: Float(guidanceScale)
            ) { cgim, p, s in
                if cgim != nil {
                    cgImage = cgim!
                    
                    image = Image(
                        cgim!,
                        scale: 1.0,
                        label: Text("Generated image")
                    )
                }
                progressProp = p
                progressStage = s
            }
            running = false
        }
    }
}

// MARK: - ContentView_Previews

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
