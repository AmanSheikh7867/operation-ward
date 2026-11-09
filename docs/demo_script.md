# Three-Minute Live Demo Storyboard & Pitch Script
### Operation Ward Zero – Intelligent Hospital-Grade Telemetry Command Center

---

## Part 1: Quick Reference Demo Controls

| Target Scenario | UI Button (Top Bar) | Keyboard / Trigger | Clinical Impact Observed |
|---|---|---|---|
| **Normal Resting** | `[🟢 Normal Resting]` | Click button | $\text{HR } 74$, $\text{SpO}_2\ 98.2\%$, $\text{RR } 16$, $\text{MAP } 92$, $\text{NEWS2 } 0$, $\text{Risk } 0.18$. Green status. |
| **Septic Shock (qSOFA)** | `[🔴 Sepsis (qSOFA)]` | Click button | $\text{HR } 124$, $\text{SpO}_2\ 91.5\%$, $\text{RR } 27$, $\text{Temp } 39.3^\circ\text{C}$, $\text{MAP } 64$, $\text{SI } 1.41$, $\text{NEWS2 } 9$, $\text{Risk } 0.92$. Alarm fires. Sepsis bundle activates. |
| **Hypoglycemic Crash** | `[⚠️ Hypoglycemia]` | Click button | $\text{Glucose } 46\text{ mg/dL}$, $\text{HR } 112$, $\text{SI } 1.08$, $\text{NEWS2 } 4$, $\text{Risk } 0.74$. Oral glucose resuscitation protocol. |
| **Respiratory Distress** | `[🔵 Respiratory Distress]` | Click button | $\text{SpO}_2\ 85\%$, $\text{RR } 32\text{ bpm}$, $\text{NEWS2 } 8$, $\text{Risk } 0.86$. High-flow oxygen titration protocol. |
| **Hypertensive Emergency** | `[🟣 Hypertensive Emergency]` | Click button | $\text{BP } 196/116\text{ mmHg}$, $\text{MAP } 142\text{ mmHg}$, suppressed HRV ($15\text{ ms}$). |
| **Dynamic Deterioration** | `[⚡ Dynamic Deterioration]` | Click button | Progressive multi-organ collapse over 60 seconds ($\text{NEWS2 } 12$, $\text{Risk } 0.98$). |

---

## Part 2: Three-Minute Pitch Script & Storyboard

### **[0:00 – 0:45] The Hook & The Problem**
- **Presenter Action**: Stand in front of the screen displaying **[http://localhost:8000](http://localhost:8000)**. Point to the top command bar and the live biometrics.
- **Spoken Script**:
  > *"Judges, over 65% of high-acuity patients discharged from hospitals are cared for at home by non-clinical family members. But current home monitoring relies on consumer smartwatches that track only basic pulse and temperature.*
  > 
  > *In clinical medicine, that is dangerous. 70% of fatal deteriorations are preceded by changes in **Respiratory Rate**, dropping **Mean Arterial Pressure (MAP)**, or surging **Shock Index**—parameters consumer gadgets cannot measure.*
  > 
  > *Welcome to **Operation Ward Zero**: an end-to-end intelligent platform that turns a home bedroom into an ICU step-down ward."*

---

### **[0:45 – 1:45] SENSE & PREDICT (Live Crisis Injection)**
- **Presenter Action**: Point to Bed 01 (currently green and stable). Then, **click `[🔴 Sepsis (qSOFA)]`** on the top scenario bar.
- **Spoken Script**:
  > *"Here on our central telemetry command center, we monitor multiple patients concurrently—Bed 01, Bed 02, and Bed 03.*
  > 
  > *Watch what happens when an acute crisis strikes Bed 01. I’ll inject a live septic shock scenario right now.*
  > 
  > *[CLICK: `[🔴 Sepsis (qSOFA)]`]*
  > 
  > *Instantly, our dual waveforms react: the Lead II ECG accelerates to 124 bpm, and our Capnography respiratory trace surges to 27 breaths per minute.*
  > 
  > *Notice the hemodynamics: Mean Arterial Pressure has collapsed to 64 mmHg—below the 65 mmHg organ perfusion threshold—while the Shock Index has surged to 1.41.*
  > 
  > *Immediately, our dual-tier engine flags the emergency: our 10-feature Machine Learning model surges to 0.92 Risk, while the hospital standard NHS NEWS2 score leaps to 9 in the High Risk emergency tier!"*

---

### **[1:45 – 2:30] UNDERSTAND & EXPLAIN (Explainable AI & Protocol)**
- **Presenter Action**: Move to the right side of the screen and highlight the **SHAP Waterfall Attribution bars** and the **Clinical Action Checklist**.
- **Spoken Script**:
  > *"Doctors and caregivers rightly reject black-box AI. Ward Zero explains every prediction.*
  > 
  > *Here in our real-time SHAP Waterfall chart, you see the exact mathematical drivers: the surging Shock Index accounts for +0.42 of the risk, abnormal glucose contributes +0.24, and tachypnea adds +0.13.*
  > 
  > *Simultaneously, the platform automatically activates the verified **Sepsis 3-Hour Bundle Action Checklist** for the caregiver: check IV access, draw lactate, and notify the intensivist.*
  > 
  > *And down in our clinical assistant workspace, our Groq AI provides immediate, non-diagnostic triage guidance in English, Hindi, and Hinglish so non-English-speaking family caregivers know exactly what to do without panicking."*

---

### **[2:30 – 3:00] RESPOND & RELIABILITY (The Close)**
- **Presenter Action**: Check off the protocol action items, click the **60-Second Trend Graph** tab, and show the CSV export.
- **Spoken Script**:
  > *"The caregiver acknowledges the alert with one click, creating an immutable, timestamped audit log.*
  > 
  > *On our 60-second multi-trace trend graph, you see the simultaneous trajectories of HR, oxygen saturation, MAP, and risk over time, with 1-click CSV export for physician review.*
  > 
  > *And even if home Wi-Fi drops, our circular in-memory retry buffer guarantees zero data loss.*
  > 
  > *Operation Ward Zero bridges the gap between hospital ICU standards and home healthcare—saving lives through Sense, Understand, Predict, Alert, and Respond. Thank you!"*
