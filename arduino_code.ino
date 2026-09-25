/*
 * Project KAI: High-Precision 512 Hz EEG Sampler
 * Microcontroller: Arduino Uno R4 Minima (32-bit RA4M1)
 * Sensor: BioAmp EXG Pill connected to Analog Pin A0
 */

const unsigned long SAMPLE_INTERVAL_US = 1953; // Target ~512 Hz (1,000,000 us / 512 = 1953.125 us)
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);      // High-speed serial transport for real-time streaming
  analogReadResolution(12);  // 12-bit ADC resolution (0 - 4095 scale)
}

void loop() {
  unsigned long currentMicros = micros();
  
  // Precise non-blocking timing interval check
  if (currentMicros - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime += SAMPLE_INTERVAL_US;
    
    int eegValue = analogRead(A0);
    Serial.println(eegValue);
  }
}
