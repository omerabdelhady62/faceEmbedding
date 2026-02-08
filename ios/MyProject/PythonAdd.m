#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(PythonAdd, NSObject)

RCT_EXTERN_METHOD(add:(nonnull NSNumber *)a
                  b:(nonnull NSNumber *)b
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end
