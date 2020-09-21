//
//  CircularBuffer.swift
//  progression-ios
//
//  Created by  Igor Peric on 21/09/2020.
//  Copyright © 2020  Igor Peric. All rights reserved.
//

import Foundation

class CircularBuffer<T: NSObject> {
    
    var storage: Array<T>!
    var currentIndex: Int = 0
    
    var capacity: Int = -1
    
    init(capacity: Int) {
        self.capacity = capacity
        self.storage = Array<T>.init(repeating: T(), count: capacity)
    }
    
    func add(element: T) {
        currentIndex = (currentIndex + 1) % capacity
        storage[currentIndex] = element
    }
    
    func getArray() -> Array<T> {
        let end = Array<T>(storage.suffix(from: currentIndex))
        let start =  Array<T>(storage.prefix(upTo: currentIndex))
        return end + start
    }
    
}
