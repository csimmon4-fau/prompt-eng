# Global prompt constants
ZERO_SHOT_PROMPT = (
    "Generate requirements for a local, "
    "privacy-preserving LLM-based tool that accurately and efficiently redacts names, "
    "emails, and other sensitive information from meeting transcripts. "
    "Ensure that requirements describe the core features of the tool, "
    "performance, security, scalability, and privacy constraints."
)

FEW_SHOT_PROMPT = (
        "The following are examples of functional and non-functional requirements for different AI-powered tools "
        "that ensure privacy and security while processing sensitive information. Use these examples to structure "
        "your response for the main task.\n\n"
        
        "### Example 1:\n"
        "**Input:**\n"
        "Design a privacy-preserving AI chatbot for financial services that protects users' personal and financial data.\n\n"
        
        "**Output:**\n"
        "**Functional Requirements:**\n"
        "1. The chatbot must detect and redact account numbers, credit card details, and personally identifiable information (PII) in real-time.\n"
        "2. Users should be able to configure custom rules for redaction based on industry-specific requirements.\n"
        "3. The system must support secure authentication and role-based access control.\n"
        "4. Chat logs should be encrypted and stored securely, with an option for automatic deletion.\n"
        "5. The chatbot must comply with financial data protection regulations like GDPR and PCI-DSS.\n\n"

        "**Non-Functional Requirements:**\n"
        "1. The system should maintain 99.99% uptime and support high availability.\n"
        "2. All communications must be end-to-end encrypted using AES-256.\n"
        "3. Redaction processing should be completed within 200ms per message to maintain responsiveness.\n"
        "4. The chatbot should be scalable to handle 10,000 concurrent users.\n"
        "5. Logs and audit trails should be securely maintained for compliance review.\n\n"

        "### Example 2:\n"
        "**Input:**\n"
        "Develop an AI-powered email filtering tool that removes sensitive personal data before emails are sent.\n\n"
        
        "**Output:**\n"
        "**Functional Requirements:**\n"
        "1. Automatically detect and redact personal identifiers such as Social Security Numbers (SSN), phone numbers, and addresses.\n"
        "2. Provide an admin dashboard to configure and review redaction policies.\n"
        "3. Allow users to add custom rules for detecting sensitive data patterns.\n"
        "4. The tool must integrate with email clients like Gmail and Outlook via API.\n"
        "5. A warning mechanism should alert users if an email contains sensitive information before sending.\n\n"

        "**Non-Functional Requirements:**\n"
        "1. The system should process emails in under 500ms to ensure smooth user experience.\n"
        "2. Redacted data should not be recoverable, ensuring full compliance with privacy regulations.\n"
        "3. The system should work offline and store data locally for security.\n"
        "4. Must comply with HIPAA and GDPR standards for data privacy.\n"
        "5. Logs of redacted information should be stored securely with restricted access.\n\n"

        "**Now, complete the following task:**\n\n"
        "**Input:**\n"      
    )

def get_prompt(prompt_type="default"):
    """
    Returns the predefined message prompt based on the given type.
    
    Args:
        prompt_type (str): Type of prompt to return. Options:
                           - "default" for basic arithmetic
                           - "zero_shot" for zero-shot learning
                           - "few_shot" for few-shot learning
    
    Returns:
        tuple: (PROMPT, model, temperature, num_ctx_tokens, num_output_tokens)
    """
    match prompt_type:
        case "zero_shot":
            return zero_shot()
        case "few_shot":
            return few_shot()
        case "self_reflective":
            return self_reflective()
        case _:
            # Default case
            MESSAGE = "1 + 1"
            model = "llama3.2:latest"  # Ensure this is a string
            temperature = 1.0
            num_ctx_tokens = 500
            num_output_tokens = 2400
            return MESSAGE, model, temperature, num_ctx_tokens, num_output_tokens

def zero_shot():
    """
    Returns a predefined zero-shot prompt with context settings.
    
    Returns:
        tuple: (prompt, model, temperature, num_ctx_tokens, num_output_tokens)
    """
    prompt = ZERO_SHOT_PROMPT
    model = "llama3.2:latest"  # Ensure this is a string
    temperature = 1.0
    num_ctx_tokens = 500
    num_output_tokens = 2400
    return prompt, model, temperature, num_ctx_tokens, num_output_tokens

def few_shot():
    """
    Returns a predefined few-shot prompt with context settings.
    
    Returns:
        tuple: (prompt, model, temperature, num_ctx_tokens, num_output_tokens)
    """
    prompt = FEW_SHOT_PROMPT + ZERO_SHOT_PROMPT
    model = "llama3.2:latest"  # Ensure this is a string
    temperature = 1.0
    num_ctx_tokens = 1500
    num_output_tokens = 5000
    return prompt, model, temperature, num_ctx_tokens, num_output_tokens

def self_reflective():
    """
    Returns a predefined self-reflective prompt with context settings.
    
    Returns:
        tuple: (refined_prompt, model, temperature, num_ctx_tokens, num_output_tokens)
    """
    model = "llama3.2:latest"  # Ensure this is a string
    temperature = 1.0
    num_ctx_tokens = 5000
    num_output_tokens = 10000
    
    prompt = ZERO_SHOT_PROMPT # just return zero shot
    return prompt, model, temperature, num_ctx_tokens, num_output_tokens

def self_reflective_prompt(prompt):
    """
    Enhances a given prompt by asking the LLM to improve it for clarity, specificity, and effectiveness.

    Args:
        prompt (str): The original prompt to be refined.

    Returns:
        str: The refined prompt.
    """
    
    SELF_REFLECTION_PROMPT = (
        "Analyze the following prompt and improve it for clarity, specificity, and effectiveness. "
        "Ensure that the revised prompt provides unambiguous instructions to an LLM and enhances "
        "the likelihood of generating a high-quality response. "
        "Additionally, maintain the original intent while making it more structured and actionable.\n\n"
        "Original Prompt:\n"
        f"{prompt}\n\n"
        "Please provide only the refined prompt below. Do not include any other commentary or explanations."
    )
    
    return SELF_REFLECTION_PROMPT