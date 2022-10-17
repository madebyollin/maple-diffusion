import SwiftUI

struct ContentView: View {
#if os(iOS)
    let mapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: true)
#else
    let mapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: false)
#endif
    let dispatchQueue = DispatchQueue(label: "Generation")
    @State var steps: Float = 20
    @State var image: Image?
    @State var prompt: String = ""
    @State var negativePrompt: String = ""
    @State var guidanceScale: Float = 7.5
    @State var running: Bool = false
    @State var progressProp: Float = 1
    @State var progressStage: String = "Ready"
    @State var latentHeight: Float = 64
    @State var latentWidth: Float = 64
    
    func loadModels() {
        dispatchQueue.async {
            running = true
            mapleDiffusion.initModels() { (p, s) -> () in
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
            mapleDiffusion.generate(prompt: prompt, negativePrompt: negativePrompt, seed: Int.random(in: 1..<Int.max), steps: Int(steps), guidanceScale: guidanceScale, latentHeight: roundedLatentHeight(), latentWidth: roundedLatentWidth()) { (cgim, p, s) -> () in
                if (cgim != nil) {
                    image = Image(cgim!, scale: 1.0, label: Text("Generated image"))
                }
                progressProp = p
                progressStage = s
            }
            running = false
        }
    }
    func roundedLatentWidth() -> Int { return Int((latentWidth / 8).rounded()) * 8 }
    func roundedLatentHeight() -> Int { return Int((latentHeight / 8).rounded()) * 8 }
    func width() -> CGFloat { return CGFloat(roundedLatentWidth() * 8) }
    func height() -> CGFloat { return CGFloat(roundedLatentHeight() * 8) }
    var body: some View {
        VStack {
#if os(iOS)
            Text("🍁 Maple Diffusion").foregroundColor(.orange).bold().frame(alignment: Alignment.center)
#endif
            if (image == nil) {
                Rectangle().fill(.gray).aspectRatio(1.0, contentMode: .fit).frame(idealWidth: width(), idealHeight: height())
            } else {
#if os(iOS)
                ShareLink(item: image!, preview: SharePreview(prompt, image: image!)) {
                    image!.resizable().aspectRatio(contentMode: .fit).frame(idealWidth: width(), idealHeight: height())
                }
#else
                image!.resizable().aspectRatio(contentMode: .fit).frame(idealWidth: width(), idealHeight: height())
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
                }.frame(width: 96, alignment: .leading)
                Slider(value: $guidanceScale, in: 1...20)
            }
            HStack {
                HStack {
                    Text("Steps").bold()
                    Text("\(Int(steps))").foregroundColor(.secondary)
                }.frame(width: 96, alignment: .leading)
                Slider(value: $steps, in: 5...150)
            }
            HStack {
                HStack {
                    Text("Latent Height").bold()
                    Text("\(Int(roundedLatentHeight()))").foregroundColor(.secondary)
                }.frame(width: 96, alignment: .leading)
                Slider(value: $latentHeight, in: 32...128)
            }
            HStack {
                HStack {
                    Text("Latent Width").bold()
                    Text("\(Int(roundedLatentWidth()))").foregroundColor(.secondary)
                }.frame(width: 96, alignment: .leading)
                Slider(value: $latentWidth, in: 32...128)
            }
            ProgressView(progressStage, value: progressProp, total: 1).opacity(running ? 1 : 0).foregroundColor(.secondary)
            Spacer(minLength: 8)
            Button(action: generate) {
                Text("Generate Image")
                    .frame(minWidth: 100, maxWidth: .infinity, minHeight: 64, alignment: .center)
                    .background(running ? .gray : .blue)
                    .foregroundColor(.white)
                    .font(Font.title)
                    .cornerRadius(32)
            }.buttonStyle(.borderless).disabled(running)
        }.padding(16).onAppear(perform: loadModels)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
