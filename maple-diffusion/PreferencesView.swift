//
//  PreferencesView.swift
//  MapleDiffusion
//
//  Created by Lilliana on 03/11/2022.
//

import SwiftUI
 
struct PreferencesView: View {
    @AppStorage("nsfw") var allowNSFWContent: Bool = false
    @AppStorage("nai") var useNovelAiPrompts: Bool = true
    @AppStorage("saving") var saveImageOnTap: Bool = true
    
    var body: some View {
        TabView {
            Group {
                VStack {
                    Toggle("Allow NSFW Content", isOn: $allowNSFWContent)
                        .padding()
                    
                    Toggle("Save Image on Tap", isOn: $saveImageOnTap)
                        .padding()
                    
                    Toggle("Use NovelAI Defaults", isOn: $useNovelAiPrompts)
                        .padding()
                }
            }
            .tabItem {
                Label("General", systemImage: "gearshape")
            }
        }
        .frame(width: 500, height: 280)
    }
}
