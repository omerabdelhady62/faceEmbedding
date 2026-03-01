import Foundation
import React

@objc(PythonAdd)
class PythonAdd: NSObject {

  @objc
  func add(_ a: NSNumber, b: NSNumber, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    // Same logic as add_numbers.py (iOS has no Python runtime)
    let sum = a.doubleValue + b.doubleValue
    resolve(NSNumber(value: sum))
  }

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}
