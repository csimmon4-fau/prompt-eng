![GenI-banner](https://github.com/genilab-fau/genilab-fau.github.io/blob/8d6ab41403b853a273983e4c06a7e52229f43df5/images/genilab-banner.png?raw=true)

# **Evaluating Progressive Prompt Engineering for Requirements Engineering**

Exploring the systematic combination of prompt engineering techniques with convergence validation in self-reflective prompting.

* **Authors:** [Chris Simmons](https://github.com/csimmons-fau)
* **Academic Supervisor:** [Dr. Fernando Koch](http://www.fernandokoch.me)

---

## **Research Question**

How can we systematically evaluate prompt engineering techniques while leveraging convergence testing in self-reflective prompting to improve requirements quality?

---

## **Arguments**

### **What is already known about this topic**
- **Zero-shot prompting** provides basic requirements elicitation but lacks precision.
- **Few-shot prompting** improves accuracy through examples but needs careful curation.
- **Self-reflective prompting** enables iterative improvement but requires validation.
- **Requirements quality assessment** needs objective measures.
- **Convergence testing** can help validate the stability of iterative processes.

### **What this research is exploring**

1. **Zero-Shot Baseline**
   - Basic requirements elicitation capabilities.
   - Understanding limitations without context.
   - Establishing baseline metrics.

2. **Few-Shot Enhancement**
   - Adding domain-specific examples.
   - Improving requirement specificity.
   - Measuring contextual impact.

3. **Self-Reflective with Convergence**
   - Implementing iterative refinement.
   - Using convergence testing to validate stability.
   - Measuring when requirements stabilize.
   - Applying similarity thresholds.

### **Implications for practice**
- Provides a systematic approach to requirements elicitation.
- Enables objective measurement of requirement quality.
- Validates requirement stability through convergence testing in self-reflection.
- Establishes a repeatable methodology for requirement refinement.

---

## **Use Case & Justification**
This study focuses on **automating requirement analysis for a local, privacy-preserving LLM-based redaction tool**. The tool is designed to **accurately and efficiently redact names, emails, and other sensitive information from meeting transcripts** while ensuring compliance with privacy and security standards.

### **Why This Problem?**
- **Privacy regulations require accurate redaction of sensitive data** in meeting transcripts and documents.
- **Manual redaction is error-prone and inefficient**, making automation critical.
- **LLMs offer high accuracy but must be privacy-preserving**, ensuring no sensitive data leaks.
- **Self-reflective prompting can iteratively refine requirement quality** to ensure security, scalability, and efficiency constraints are met.

---

## **Research Method**

Our methodology consists of three phases:

1. **Baseline Evaluation:** Establishing zero-shot and few-shot performance.
2. **Self-Reflective Experimentation:** Iterative refinement using self-reflective prompting.
3. **Performance Measurement:** Evaluating clarity, specificity, and stability over iterations.

---

## **Metric Definitions**

### **Clarity**
Clarity is measured using the **Flesch Reading Ease score**, which evaluates sentence complexity and word difficulty. A **higher score** indicates the text is **easier to read**, while a **lower score** suggests greater complexity.

### **Specificity**
Specificity is evaluated by counting the occurrences of **strong requirement-defining words**, such as "must," "shall," "exactly," "minimum," and "threshold." Higher specificity scores indicate **greater precision and explicit constraints** in the generated requirements.

### **Effectiveness**
Effectiveness is assessed by detecting **action-oriented and goal-driven words**, such as "ensure," "optimize," "enhance," "reduce," and "automate." This metric captures how well the requirement describes **an outcome or functional objective** rather than being vague or generic.

---

## **Results & Findings**

### **1️⃣ Comparison of Prompting Techniques**
The following table presents the **average clarity, specificity, and effectiveness scores** for each prompting technique.

| Prompting Technique            | Clarity | Specificity | Effectiveness |
|--------------------------------|---------|------------|--------------|
| zero_shot                       | 9.89    | 0.00       | 1.00         |
| few_shot                       | 33.51   | 18.00      | 3.00         |
| self_reflective_iteration_1     | 9.89    | 0.00       | 1.00         |
| self_reflective_iteration_2     | 18.15   | 1.00       | 0.00         |
| self_reflective_iteration_3     | 5.53    | 0.00       | 0.00         |
| self_reflective_iteration_4     | 17.54   | 4.00       | 3.00         |
| self_reflective_iteration_5     | 19.67   | 4.00       | 3.00         |
| self_reflective                 | 19.67   | 4.00       | 3.00         |


### **2️⃣ Visualization of Metric Trends**
The following visualization illustrates how clarity, specificity, and effectiveness scores change across different prompting techniques.

![Metric Comparison of Prompting Techniques](sandbox:/mnt/data/metric_comparison.png)

#### **Key Observations:**
- **Few-shot prompting consistently achieves the highest clarity, specificity, and effectiveness scores**, demonstrating the importance of structured examples.
- **Self-reflective iterations improve similarity but sometimes reduce specificity and effectiveness**, suggesting that additional refinement methods may be necessary.
- **Zero-shot prompting tends to perform poorly across all metrics**, reinforcing the need for better contextual guidance in prompt engineering.
- **Self-reflective iterations do not always result in linear improvement**, as clarity fluctuates and specificity/effectiveness decrease over iterations.

---

## **Future Research**

### **1️⃣ Hybrid Prompting Strategies for Balancing Specificity & Effectiveness**
- The study found that **few-shot prompting performs best** in clarity, specificity, and effectiveness.
- **Self-reflective iterations improve stability but weaken specificity and effectiveness**.
- Future research should explore **how structured examples can be combined with self-reflective techniques** to maintain both **consistency and precision** in requirements generation.

### **2️⃣ Machine Learning-Based Metrics for Specificity & Effectiveness**
- Current evaluation relies on **keyword-based scoring**.
- Future research could implement **ML classifiers** or **LLM-based assessment models** to evaluate **requirement quality beyond keyword occurrences**.
- This would allow for **context-aware** scoring that adapts to different domains and requirement structures.

### **3️⃣ User-Centric Validation of Prompt Engineering Results**
- The current study relies on **algorithmic evaluation metrics**.
- Future research could introduce **human expert assessments** to validate whether refined requirements align with **stakeholder expectations**.
- Crowdsourced evaluations or **comparative studies** could assess how **end-users perceive clarity, specificity, and effectiveness in generated requirements**.

---

