//
//  ViewController.swift
//  progression-ios
//
//  Created by  Igor Peric on 20/09/2020.
//  Copyright © 2020  Igor Peric. All rights reserved.
//

import UIKit
import AVKit
import CoreML

class ViewController: UIViewController {

    var chordDetection: ChordDetectionModel = ChordDetectionModel()
    
    private var audioEngine: AVAudioEngine!
    private var mic: AVAudioInputNode!
    
    var audioInput: MLMultiArray!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        configureAudioSession()
        audioEngine = AVAudioEngine()
        mic = audioEngine.inputNode
        
        do {
            audioInput = try MLMultiArray.init(shape: [1, 48000, 1], dataType: .float32)
        } catch {
            
        }
        
        let micFormat = mic.inputFormat(forBus: 0)
        print(micFormat.sampleRate) // 48000
        print(micFormat.channelCount) // 1
        mic.installTap(onBus: 0, bufferSize: audioInput.shape[1].uint32Value, format: micFormat) { (buffer, when) in
            let sampleData = UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
            for (i, v) in sampleData.enumerated() {
                self.audioInput[i] = NSNumber(value: v)
            }
            let input = ChordDetectionModelInput(waveform: self.audioInput)
            do {
                let prediction = try self.chordDetection.prediction(input: input)
                print(prediction.featureValue(for: "Identity")!)
            } catch {
                
            }
        }
        startEngine()
    }

    private func startEngine() {
        guard !audioEngine.isRunning else {
            return
        }

        do {
            try audioEngine.start()
        } catch { }
    }
    
    private func configureAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(AVAudioSession.Category.record, options: [.mixWithOthers])
            try AVAudioSession.sharedInstance().setActive(true)
        } catch { }
    }

}

