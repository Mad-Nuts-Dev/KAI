void setup() {
  Serial.begin(115200);  // Fast serial for real-time EEG
  analogReadResolution(12);  // 12-bit resolution for better accuracy (0-4095)
}

void loop() {
  int eegValue = analogRead(A0);
  Serial.println(eegValue);
  delay(5);  // Adjust based on how much data you want to send (5ms = 200Hz)
}
