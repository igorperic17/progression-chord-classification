//
//  Extensions.swift
//  progression-ios
//
//  Created by  Igor Peric on 21/09/2020.
//  Copyright © 2020  Igor Peric. All rights reserved.
//

import Foundation

extension Array where Element: Comparable {
  /**
    Returns the index and value of the largest element in the array.
    - Note: This method is slow. For faster results, use the standalone
            version of argmax() instead.
  */
  public func argmax() -> (Int, Element) {
    precondition(self.count > 0)
    var maxIndex = 0
    var maxValue = self[0]
    for i in 1..<self.count where self[i] > maxValue {
      maxValue = self[i]
      maxIndex = i
    }
    return (maxIndex, maxValue)
  }
}
