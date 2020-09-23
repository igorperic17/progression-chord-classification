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
    
    @IBOutlet weak var chordNameLabel: UILabel!
    
    var audioInput: MLMultiArray!
    
    let operationQueue: OperationQueue = OperationQueue()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        operationQueue.maxConcurrentOperationCount = 1
        
        configureAudioSession()
        audioEngine = AVAudioEngine()
        mic = audioEngine.inputNode
        let micFormat = mic.inputFormat(forBus: 0)
        let targetFormat = AVAudioFormat.init(commonFormat: AVAudioCommonFormat.pcmFormatFloat32, sampleRate: 48000, channels: 1, interleaved: false)
        
//        // handle audio piping
//        // our new node
//        let DJ = AVAudioMixerNode()
//
//        // attach the new node to the audio engine
//        audioEngine.attach(DJ)
//
//        // connect input to the new node, using the input's format
//        audioEngine.connect(mic, to: DJ, format: micFormat)
//        // connect the new node to the output node
////        audioEngine.connect(DJ, to: audioEngine.outputNode, format: targetFormat)
        
        let featureVectorSize = chordDetection.model.modelDescription.inputDescriptionsByName["waveform"]!.multiArrayConstraint!.shape[1]
        do {
            audioInput = try MLMultiArray.init(shape: [1, featureVectorSize, 1], dataType: .float32)
        } catch {
            
        }
        
        let circularBuffer = CircularBuffer<NSNumber>(capacity: Int(featureVectorSize))
        let sampleSubsampling = 10
        var subsamplingCounter = 0
        
        var classes = [ "No", "Em", "Cmaj", "Amaj" ]
        classes.sort()
        
        // tap on the new node
        mic.installTap(onBus: 0, bufferSize: 1000, format: targetFormat, block:
                { (buffer: AVAudioPCMBuffer!, time: AVAudioTime!) -> Void in
                    let sampleData = UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
                    
                    for (_, v) in sampleData.enumerated() {
                        subsamplingCounter = (subsamplingCounter + 1) % sampleSubsampling
                        if subsamplingCounter == 0 {
                            circularBuffer.add(element: NSNumber(value: v))
                            
//                            if self.operationQueue.operationCount > 0 { return }
                            
                            self.operationQueue.addOperation {
                                
                                let audioData = circularBuffer.getArray()
                                for (i, v) in audioData.enumerated() {
                                    self.audioInput[i] = v
                                }
    //                            print(self.audioInput[0].floatValue)
                                
                                do {
                                    let prediction = try self.chordDetection.prediction(waveform: self.audioInput)
                                    if let probabilities = prediction.featureValue(for: "Identity")!.multiArrayValue {
                                        var maxIdx = -1
                                        for i in 0...probabilities.count {
                                            if maxIdx == -1 || probabilities[i].floatValue > probabilities[maxIdx].floatValue {
                                                maxIdx = i
                                            }
                                        }
                                        let label = classes[maxIdx]
//                                        print(label)
                                        
                                        DispatchQueue.main.async {
                                            self.chordNameLabel.text = label
                                        }
                                    }
                                } catch {
                                    
                                }
                            }
                        }
                    }
        })
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

