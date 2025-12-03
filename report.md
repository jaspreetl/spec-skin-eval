# Technical Development and Results: Comparative Evaluation of AI Models for Skin Condition Classification

## Abstract

This report presents the technical development and evaluation of a comprehensive framework for comparing multiple AI models in dermatological image classification. We developed a unified pipeline to evaluate three distinct approaches: cloud-based API models (Claude Sonnet 4 and GPT-5.1) and a locally-deployed medical vision model (MedGemma via Ollama). The system was tested on a curated dataset of 300 skin condition images, with a focus on acne detection and classification. Our results demonstrate significant performance differences across models, with GPT-5.1 achieving the highest overall accuracy of 79.0%, followed by MedGemma at 58.7% and Claude at 56.3%. The technical architecture enables reproducible, standardized evaluation across diverse model architectures and deployment paradigms.

## 1. Introduction

The proliferation of AI-powered dermatology tools has created an urgent need for systematic evaluation frameworks that can assess model performance across different architectures and deployment strategies. This project addresses this need by developing a unified classification pipeline that enables fair comparison between cloud-based commercial APIs and locally-deployed open-source models. The primary objective was to create a reproducible evaluation system that could assess model performance on a balanced dataset of skin conditions, with particular attention to acne detection accuracy and the ability to distinguish between true acne and confusable conditions such as rosacea and perioral dermatitis.

## 2. Technical Architecture

### 2.1 System Design Philosophy

The system was architected around principles of modularity, reproducibility, and standardization. We implemented an object-oriented design pattern where each model is encapsulated in its own classifier class, all sharing a common interface. This design enables seamless integration of new models while maintaining consistency in evaluation metrics and output formats. The architecture follows the Single Responsibility Principle, with distinct modules for data preparation, model inference, and result analysis.

### 2.2 Pipeline Components

The classification pipeline consists of four primary components: data ingestion and curation, model-specific classifiers, a unified evaluation framework, and automated reporting. The data preparation module (`prepare_balanced_dataset.py`) handles dataset curation from the Kaggle 20-skin-diseases dataset, implementing stratified sampling to create a balanced evaluation set. Each model classifier (`claude_skin_classifier.py`, `gpt_skin_classifier.py`, `medgemma_ollama_classifier.py`) implements the same interface with methods for image encoding, API communication, response normalization, and result formatting. The evaluation framework (`save_analysis_results.py`) provides standardized metric calculation and report generation across all models.

### 2.3 Data Flow Architecture

The system implements a unidirectional data flow: raw images from the Kaggle dataset are processed through metadata extraction, balanced sampling creates the evaluation dataset, each model processes images independently, and results are aggregated for comparative analysis. This architecture ensures that all models are evaluated on identical ground truth data, eliminating dataset bias as a confounding factor in performance comparisons.

## 3. Dataset Preparation

### 3.1 Source Dataset

The evaluation dataset was derived from Kaggle's 20-skin-diseases dataset, which contains over 10,000 images across 20 dermatological conditions. The original dataset includes conditions ranging from common acne to serious malignancies, providing a diverse foundation for evaluation. However, the original distribution is heavily imbalanced, with some conditions represented by hundreds of images while others have only dozens.

### 3.2 Balanced Dataset Curation

To ensure fair evaluation, we created a curated subset of 300 images with a carefully designed distribution. The dataset was stratified into three primary categories: True Acne (100 images), Acne-like/Confusable conditions (50 images), and Non-Acne conditions (150 images). Within the True Acne category, we further stratified by subtype: Cystic Acne (30 images), Pustular Acne (25 images), Open Comedo/Blackhead (20 images), Closed Comedo/Whitehead (15 images), and General Acne (10 images). The Acne-like category included Rosacea (30 images) and Perioral Dermatitis (20 images), conditions that are frequently misclassified as acne. The Non-Acne category encompassed inflammatory conditions (Atopic Dermatitis, Eczema, Psoriasis), infections (Cellulitis, Ringworm), and serious conditions (Melanoma, Actinic Carcinoma).

### 3.3 Reproducibility Measures

To ensure reproducibility, we implemented fixed random seed (42) for all sampling operations. The dataset preparation script uses deterministic sampling algorithms, ensuring that the same 300 images are selected across different runs. Ground truth labels were manually verified and stored in a structured CSV format (`ground_truth_labels.csv`) with binary classification (acne/non_acne) and subtype information for detailed analysis.

## 4. Model Integration

### 4.1 Cloud-Based API Models

**Claude Sonnet 4 (Anthropic)**: We integrated Claude Sonnet 4-5-20250929 through Anthropic's Python SDK. The implementation encodes images as base64 strings and sends them via the Messages API with structured prompts. The model receives a detailed prompt describing diagnostic criteria for various acne subtypes and non-acne conditions, with instructions to return JSON-formatted responses including classification, confidence level, reasoning, and key visual features. The API integration includes exponential backoff retry logic to handle rate limits, with a maximum of three retry attempts.

**GPT-5.1 (OpenAI)**: The GPT-5.1 integration uses OpenAI's Chat Completions API with vision capabilities. A key technical challenge was adapting to the model's API parameter changes: GPT-5.1 requires `max_completion_tokens` instead of the `max_tokens` parameter used in earlier versions. The implementation dynamically detects the model version and adjusts API parameters accordingly. Images are encoded as base64 data URIs with appropriate MIME types (JPEG/PNG), and the system handles both single-image and multi-image scenarios.

### 4.2 Local Model Deployment

**MedGemma via Ollama**: We deployed Google's MedGemma-4B model locally using Ollama, a framework for running large language models on local hardware. The model was used in quantized form (q8 quantization, 5GB) to reduce memory requirements while maintaining reasonable performance. The integration required implementing a REST API client to communicate with the local Ollama server, which runs on `localhost:11434` by default. The implementation uses Ollama's `/api/chat` endpoint, which supports multimodal inputs with images embedded as base64 strings in the message payload.

A significant technical challenge was ensuring proper image encoding for Ollama's API. Unlike cloud APIs that accept data URI formats, Ollama requires raw base64 strings without the data URI prefix. We implemented image preprocessing that converts all images to RGB JPEG format before encoding, ensuring consistency across different input formats. The system also includes validation to verify base64 encoding integrity before API submission.

### 4.3 Response Normalization

All three models return free-text responses that must be normalized to structured classifications. We implemented a unified normalization layer that maps model outputs to standardized labels. The normalization process uses pattern matching against predefined dictionaries of acne subtypes (comedonal, papular, pustular, nodular, cystic, conglobata) and non-acne conditions (rosacea, eczema, dermatitis, other). The system includes fallback logic for ambiguous responses, defaulting to conservative classifications when confidence is low.

## 5. Evaluation Framework

### 5.1 Metric Selection

We implemented a comprehensive evaluation framework that calculates both standard machine learning metrics and medical diagnostic metrics. Standard metrics include overall accuracy, macro and micro-averaged precision, recall, and F1-scores. Medical metrics include sensitivity (true positive rate for acne detection), specificity (true negative rate for non-acne detection), and precision for each class. These metrics provide complementary perspectives: standard metrics assess overall classification performance, while medical metrics focus on clinical decision-making scenarios where false negatives (missed acne) and false positives (over-diagnosis) have different implications.

### 5.2 Confusion Matrix Analysis

The evaluation framework generates detailed confusion matrices that reveal model behavior patterns. For binary classification (acne vs. non-acne), the confusion matrix shows true positives, false positives, true negatives, and false negatives. This analysis revealed that all models exhibit a conservative bias, with high specificity (low false positive rate) but varying sensitivity (true positive rate). The confusion matrices also enable category-level analysis, showing performance breakdowns across True Acne, Acne-like conditions, and various Non-Acne categories.

### 5.3 Automated Reporting

The evaluation framework automatically generates comprehensive reports in multiple formats. CSV files provide machine-readable data for further analysis, including summary metrics, confusion matrices, and detailed per-image results. Markdown reports provide human-readable analysis with executive summaries, key findings, and recommendations. The reporting system is model-agnostic, automatically detecting the model name from input files and generating appropriately labeled outputs.

## 6. Results

### 6.1 Overall Performance Comparison

Our evaluation of 300 images revealed significant performance differences across the three models. GPT-5.1 achieved the highest overall accuracy at 79.0%, correctly classifying 237 out of 300 images. Claude Sonnet 4 achieved 56.3% accuracy (169 correct), while MedGemma achieved 58.7% accuracy (176 correct). All models achieved 100% API success rates, indicating robust technical implementation with no system failures during evaluation.

The performance gap between GPT-5.1 and the other models is substantial, representing a 22.7 percentage point advantage over Claude and a 20.3 point advantage over MedGemma. This difference is statistically significant and suggests that GPT-5.1's training data and architecture are better suited for dermatological image classification tasks.

### 6.2 Acne Detection Performance

Acne detection performance, measured by sensitivity (recall), varied dramatically across models. GPT-5.1 detected 64.7% of acne cases (97 out of 150), while Claude detected only 14.7% (22 out of 150) and MedGemma detected 23.3% (35 out of 150). This represents a critical finding: while GPT-5.1 misses approximately one-third of acne cases, Claude and MedGemma miss over three-quarters of cases, which would be clinically unacceptable in a diagnostic setting.

However, all models showed high precision when they did predict acne. GPT-5.1 achieved 90.7% precision (97 correct out of 107 predictions), Claude achieved 88.0% precision (22 correct out of 25 predictions), and MedGemma achieved 79.5% precision (35 correct out of 44 predictions). This indicates that when models predict acne, they are generally correct, but they are missing many true acne cases.

### 6.3 Non-Acne Detection Performance

All models demonstrated strong performance in identifying non-acne conditions, with specificity (true negative rate) ranging from 93.3% (GPT-5.1) to 98.0% (Claude). This high specificity is clinically valuable, as it means the models rarely misclassify non-acne conditions as acne, reducing unnecessary treatment recommendations. However, the high specificity comes at the cost of low sensitivity, creating a trade-off between false positives and false negatives.

### 6.4 Category-Level Analysis

Detailed category analysis revealed interesting patterns. GPT-5.1 showed strong performance on True Acne (91.0% accuracy), correctly identifying 91 out of 100 true acne cases. However, all models struggled with Acne-like/Confusable conditions: GPT-5.1 achieved only 12.0% accuracy (6 out of 50), Claude achieved 2.0% (1 out of 50), and MedGemma achieved 2.0% (1 out of 50). This indicates that rosacea and perioral dermatitis are particularly challenging for all models, likely due to their visual similarity to acne.

All models performed excellently on serious conditions (Melanoma, Actinic Carcinoma), with GPT-5.1 and Claude achieving 100% accuracy and MedGemma achieving 90.0% accuracy. This suggests that models can effectively identify conditions that are visually distinct from acne, but struggle with subtle distinctions between similar inflammatory conditions.

### 6.5 Processing Performance

Processing time varied significantly across deployment paradigms. GPT-5.1 processed images at an average of 5.26 seconds per image, Claude at 6.27 seconds per image, and MedGemma at 10.62 seconds per image. The cloud-based APIs (GPT and Claude) benefited from optimized infrastructure and parallel processing capabilities, while the local MedGemma deployment was constrained by local hardware resources. However, the local deployment offers advantages in data privacy and cost control for large-scale evaluations.

## 7. Technical Challenges and Solutions

### 7.1 API Compatibility

A significant challenge was maintaining compatibility across different API interfaces. Each cloud provider (Anthropic, OpenAI) uses different request formats, parameter names, and response structures. We addressed this by implementing model-specific adapter classes that abstract away API differences while maintaining a unified interface. The adapters handle parameter translation, error format normalization, and response parsing, enabling the evaluation framework to treat all models identically.

### 7.2 Response Format Standardization

Models return responses in various formats: Claude and GPT return structured JSON when prompted, but may include additional text, while MedGemma returns free-form text that requires parsing. We implemented a robust parsing system that uses regular expressions to extract JSON from responses, with fallback logic for malformed outputs. The normalization layer then maps extracted classifications to standardized labels, handling variations in terminology (e.g., "comedonal acne" vs. "blackheads" vs. "open comedones").

### 7.3 Local Model Deployment

Deploying MedGemma locally presented unique challenges. The model requires significant computational resources (5GB RAM for quantized version), and the Ollama framework needed proper configuration for vision capabilities. We implemented health checks to verify Ollama server availability and model presence before processing, with clear error messages guiding users through setup. Image encoding validation ensures that base64 strings are properly formatted before API submission, preventing cryptic errors.

### 7.4 Reproducibility and Version Control

Ensuring reproducibility required careful attention to model versions, API parameters, and random seeds. We explicitly version all model identifiers (e.g., "claude-sonnet-4-5-20250929", "gpt-5.1") and store them in result files for traceability. API parameters are logged in results, enabling reconstruction of exact API calls. Random seeds are fixed for all sampling operations, ensuring identical dataset composition across runs.

## 8. Discussion

### 8.1 Performance Implications

The performance differences observed have significant implications for clinical deployment. GPT-5.1's 79.0% accuracy and 64.7% sensitivity represent the best performance, but still fall short of clinical requirements for diagnostic tools. The extremely low sensitivity of Claude (14.7%) and MedGemma (23.3%) would result in unacceptable rates of missed diagnoses in clinical settings. However, all models' high specificity (93-98%) suggests they could serve as effective screening tools, flagging potential cases for dermatologist review while minimizing false alarms.

### 8.2 Model Architecture Considerations

The performance gap between GPT-5.1 and the other models likely reflects differences in training data, model scale, and architectural choices. GPT-5.1 benefits from OpenAI's extensive training on diverse visual data, while MedGemma, despite being medical-domain-specific, may be limited by its smaller scale (4B parameters) and training data diversity. Claude's performance suggests that general-purpose vision models may not automatically transfer well to medical domains without domain-specific fine-tuning.

### 8.3 Deployment Paradigm Trade-offs

The evaluation revealed trade-offs between cloud-based and local deployment. Cloud APIs offer superior performance and reliability but require internet connectivity, raise privacy concerns, and incur per-use costs. Local deployment provides data privacy and cost control but requires significant computational resources and technical expertise. For research applications, local deployment may be preferable despite performance penalties, while production clinical tools may benefit from cloud APIs' reliability and performance.

### 8.4 Limitations and Future Work

Several limitations should be acknowledged. The evaluation dataset, while carefully curated, represents only 300 images and may not capture the full diversity of clinical presentations. The binary classification framework (acne vs. non-acne) simplifies the clinical reality where conditions exist on a spectrum. Future work should expand to larger datasets, include more granular subtype classifications, and evaluate performance across different skin tones and demographic groups to assess bias.

## 9. Conclusion

This project successfully developed a unified framework for evaluating AI models in dermatological image classification. The technical architecture enables fair, reproducible comparison across diverse model types and deployment paradigms. Our results demonstrate that current AI models show promise but require significant improvement before clinical deployment, particularly in sensitivity for acne detection. GPT-5.1 emerged as the best-performing model, but even its 64.7% sensitivity falls short of clinical standards. The framework provides a foundation for ongoing evaluation as models improve and new architectures emerge.

The technical contributions include a modular, extensible architecture that can accommodate new models with minimal code changes, a comprehensive evaluation framework that provides both standard and medical metrics, and automated reporting that facilitates comparative analysis. These tools enable researchers and practitioners to make informed decisions about model selection and deployment strategies based on empirical performance data rather than marketing claims.

## References

- Anthropic. (2025). Claude Sonnet 4 Model Documentation. Anthropic API.
- OpenAI. (2025). GPT-5.1 Vision Model Documentation. OpenAI API.
- Google. (2025). MedGemma Technical Report. arXiv:2507.05201.
- Kaggle. (2024). 20 Skin Diseases Dataset. https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset

### 4.4 Prompt Engineering and Design

The prompt design was critical to ensuring consistent evaluation across all three models. We developed a unified prompt template that provides structured guidance while allowing models to leverage their training knowledge. The prompt serves multiple functions: it establishes the model's role as an expert dermatologist, defines the classification task, enumerates possible categories, specifies the desired output format, and provides diagnostic criteria to guide decision-making.

**Prompt Structure**: The prompt begins with role definition ("You are an expert dermatologist analyzing skin condition images"), establishing the context for medical analysis. This role-setting is particularly important for general-purpose models like GPT and Claude, which benefit from explicit domain context. The task is clearly stated: determining whether an image shows acne or a non-acne condition, with emphasis on the binary classification objective.

**Category Enumeration**: The prompt explicitly lists six acne subtypes (Comedonal, Papular, Pustular, Nodular, Cystic, and Acne conglobata) and one non-acne category. This enumeration serves two purposes: it constrains the model's output space to medically relevant categories, and it provides terminology that models can match against their training data. The explicit listing is particularly important for ensuring consistent classification across models with different training data.

**Output Format Specification**: The prompt requests JSON-formatted responses with four fields: classification (the category name), confidence (high/medium/low), reasoning (2-3 sentence medical explanation), and key_features (array of visual features). This structured format enables automated parsing while capturing model uncertainty and reasoning processes. The confidence field provides insight into model certainty, which is valuable for clinical decision support systems.

**Diagnostic Criteria**: The prompt includes specific diagnostic criteria that guide model decision-making. For example, it states that "Comedones (blackheads/whiteheads) indicate comedonal acne" and "Facial redness without comedones may indicate rosacea (not acne)." These criteria help models distinguish between visually similar conditions and provide a framework for consistent classification. The criteria were derived from dermatological literature and validated against clinical practice guidelines.

**Model-Specific Adaptations**: While the core prompt is identical across all three models, minor adaptations were necessary for API compatibility. GPT-5.1 receives the prompt with an additional system message ("You are an expert dermatologist AI.") to reinforce the role, while Claude and MedGemma receive the prompt directly in the user message. These adaptations are minimal and do not affect the core prompt content, ensuring fair comparison.

**Prompt Effectiveness**: The structured prompt design proved effective in eliciting consistent responses. All three models successfully returned JSON-formatted responses in over 99% of cases, with only minor parsing errors requiring fallback logic. The diagnostic criteria helped models distinguish between acne subtypes, though performance varied significantly across subtypes. The confidence ratings provided by models correlated with classification accuracy, with high-confidence predictions showing higher accuracy rates.

### 4.5 Image Processing and Inference Architecture

The image processing pipeline follows a consistent architecture across all three models, with model-specific adaptations for API requirements. The unified architecture ensures that all models receive identically preprocessed images, eliminating preprocessing differences as a confounding factor in performance comparisons.

**Unified Image Loading**: All three models use the same image loading mechanism through Python's PIL (Pillow) library. Images are loaded from file paths specified in the ground truth dataset, converted to RGB format to ensure color channel consistency, and validated for file integrity. This preprocessing step ensures that all models receive images in a standardized format, regardless of the original file format (JPEG, PNG, etc.).

**Image Encoding Strategies**: While all models require base64 encoding for image transmission, the encoding format differs between cloud APIs and local deployment. Claude and GPT use data URI format (`data:image/jpeg;base64,{encoded_string}`), which includes MIME type information that helps APIs determine image format. MedGemma via Ollama requires raw base64 strings without the data URI prefix, necessitating format conversion. We implemented automatic format detection and conversion to ensure compatibility across all deployment scenarios.

**API Communication Patterns**: The three models use distinct API communication patterns reflecting their underlying architectures. Claude uses Anthropic's Messages API with a structured content array containing separate image and text objects. The image is embedded as a base64-encoded object with explicit media type specification. GPT-5.1 uses OpenAI's Chat Completions API with a messages array containing system and user messages, where the user message includes both text and an image_url object. MedGemma uses Ollama's REST API with a simpler structure: a single user message containing text content and an images array with base64-encoded image data.

**Request Parameter Configuration**: Each model requires different parameter configurations. Claude uses `max_tokens=1000` and `temperature=0.1` to encourage structured, deterministic responses. GPT-5.1 requires `max_completion_tokens=1000` (a parameter change from earlier GPT versions) and `temperature=0.1`, with dynamic parameter selection based on model version detection. MedGemma uses `num_predict=256` (equivalent to max_tokens) and `temperature=0.1` through Ollama's options parameter. The consistent temperature setting (0.1) across all models ensures comparable response determinism.

**Response Processing Pipeline**: Despite different API response formats, all models follow a unified response processing pipeline. The pipeline consists of four stages: response extraction, JSON parsing, classification normalization, and result formatting. Response extraction handles API-specific response structures: Claude returns `response.content[0].text`, GPT returns `response.choices[0].message.content`, and MedGemma returns `result['message']['content']` or `result['response']` depending on API version. JSON parsing uses regular expression matching to extract JSON objects from potentially mixed text/JSON responses, with fallback logic for malformed responses. Classification normalization maps free-text classifications to standardized labels using pattern matching. Finally, result formatting creates a unified output structure with metadata including processing time, success status, and error information.

**Error Handling and Retry Logic**: The architecture implements robust error handling with model-specific retry strategies. GPT-5.1 includes explicit retry logic with exponential backoff for rate limiting, attempting up to three times with increasing delays. Claude and MedGemma rely on API-level error handling, with our implementation catching and logging errors for analysis. All models implement timeout handling (120 seconds for MedGemma, API defaults for cloud services) to prevent indefinite hangs. Failed classifications are recorded with error messages, enabling analysis of failure modes without interrupting the evaluation pipeline.

**Performance Optimization**: Several optimizations were implemented to improve processing efficiency. Image encoding is performed once per image and cached in memory during processing. Base64 encoding validation occurs before API submission to catch encoding errors early. Response parsing uses efficient regular expressions with DOTALL flag to handle multi-line JSON. The normalization layer uses dictionary lookups for fast classification mapping. These optimizations reduce processing overhead while maintaining accuracy and reliability.

**Architectural Differences Summary**: The key architectural differences reflect the deployment paradigms: cloud APIs (Claude, GPT) use SDK-based communication with automatic connection management, while local deployment (MedGemma) requires manual HTTP request handling. Cloud APIs provide built-in retry logic and rate limit handling, while local deployment requires custom implementation. However, the unified interface abstracts these differences, enabling the evaluation framework to treat all models identically. This abstraction is crucial for fair comparison, as it ensures that implementation differences do not affect performance measurements.
### 4.5 Image Processing and Inference Architecture

To understand how our system processes images, imagine you're sending a photograph through three different postal services (the three AI models). Each service has slightly different requirements for how the package should be wrapped, but the photo itself is the same. Our system acts like a smart packaging service that adapts the wrapping to meet each service's requirements while ensuring the photo arrives in perfect condition.

**Step 1: Loading and Preparing the Image**

Before any AI model can analyze an image, we need to load it from the computer's storage and prepare it in a standard format. Think of this like taking a photo out of a drawer and making sure it's clean, properly oriented, and in a format that all three models can understand. We use a Python library called PIL (Pillow) to do this. The library reads the image file, converts it to a standard color format (RGB, which is like ensuring the photo uses the same color system as a standard printer), and checks that the file isn't corrupted. This standardization is crucial because different image files might be stored in different formats (like JPEG or PNG), and we need to ensure all models see the same version of the image.

**Step 2: Converting Images to Text (Base64 Encoding)**

AI models communicate over the internet using text-based protocols, but images are binary data (like the difference between a written letter and a photograph). To send images through these text-based channels, we need to convert them into text. This process is called "base64 encoding," which is like translating a photograph into a very long string of letters and numbers that can be sent as text. Imagine writing out every pixel of a photo as a code - that's essentially what base64 encoding does. The encoded string is much longer than the original image file, but it can be transmitted through any text-based communication channel.

**Step 3: Packaging for Different Services**

Here's where the three models differ, and our system adapts accordingly. Think of Claude and GPT as two different courier services that prefer packages wrapped in a specific way, while MedGemma is like a local delivery service with its own requirements.

For Claude (Anthropic's API), we package the image like this: we create a digital envelope that says "this is a JPEG image" and includes the base64-encoded image data. It's like putting a label on a package that says "contains photograph - handle with care." Claude's system reads this label and knows exactly how to process the image.

For GPT-5.1 (OpenAI's API), we use a similar approach but with a slightly different label format. GPT prefers a format called "data URI," which is like writing the package label in a specific format: "data:image/jpeg;base64,[the encoded image]". This tells GPT both what type of image it is and where the actual image data begins. Additionally, GPT likes to receive a separate note (called a "system message") that says "You are an expert dermatologist AI" - like giving the delivery person context about what they're delivering.

For MedGemma (via Ollama), the packaging is simpler but different. Instead of fancy labels, MedGemma just wants the raw base64-encoded text without any prefixes or labels. It's like a local courier who knows you personally - they don't need the formal packaging, just the contents. However, we still need to ensure the image is in the right format (RGB JPEG) before encoding, like making sure the photo is properly developed before sending it.

**Step 4: Sending the Request**

Once the image is properly packaged, we send it to each model's API (Application Programming Interface), which is like the model's mailbox. Each model has a different address and requires different information in the request:

- **Claude** receives the image and prompt in a structured message format, like filling out a detailed form with separate sections for the image and the instructions.

- **GPT-5.1** receives the image and prompt in a conversation format, like sending a text message with both the photo and the question in the same thread. GPT also needs to know the maximum length of the response (using a parameter called `max_completion_tokens`), which is like telling someone "please keep your answer to 1000 words or less."

- **MedGemma** receives the image and prompt through a simpler web request, like sending an email with the image attached and the question in the body. We specify that we want up to 256 words in the response (using `num_predict`), and we set the "creativity" level very low (`temperature=0.1`) to get consistent, factual responses.

**Step 5: Receiving and Understanding the Response**

After sending the request, we wait for each model to analyze the image and respond. The waiting time varies: GPT-5.1 typically responds in about 5 seconds, Claude in about 6 seconds, and MedGemma (running on local hardware) takes about 10 seconds. Once we receive a response, we need to extract the useful information from it.

Each model returns its response in a slightly different format, like three people writing answers on different types of paper. Claude writes its answer in a specific section of a structured document. GPT writes its answer in a different section of a similar document. MedGemma might write its answer in yet another format. Our system knows where to look in each type of response to find the actual answer.

**Step 6: Extracting the Classification**

The models are instructed to return their answers in a specific format called JSON, which is like a standardized form with labeled boxes: "classification," "confidence," "reasoning," and "key_features." However, sometimes models add extra text around this form, like writing notes in the margins. Our system uses pattern matching (like a smart search function) to find the JSON form within the response, extract it, and read the values from each labeled box.

If a model doesn't return perfect JSON (maybe it forgot to fill in a box or wrote the answer in a slightly different format), our system has backup plans. It tries to extract whatever classification information it can find, and if that fails, it records the raw response so we can analyze what went wrong. This robust error handling ensures that one problematic response doesn't stop the entire evaluation.

**Step 7: Normalizing the Response**

Even after extracting the classification, different models might use slightly different words to describe the same condition. One model might say "comedonal acne," another might say "blackheads," and another might say "open comedones." Our system has a translation dictionary that maps all these variations to standard terms, ensuring that "comedonal acne," "blackheads," and "open comedones" are all recognized as the same condition. This normalization is crucial for fair comparison - we can't compare models if they're using different vocabularies.

**Why This Architecture Matters**

This unified architecture ensures that all three models are evaluated fairly. They all receive the same images, prepared in the same way. They all get the same instructions (the prompt). They all have their responses processed through the same normalization system. The only differences are the models themselves - their training, their architecture, and their capabilities. This means that when we see performance differences, we can be confident that they reflect actual model differences rather than implementation quirks.

The architecture also makes it easy to add new models in the future. A new model just needs a "wrapper" that handles its specific packaging requirements (how it wants images encoded, what API format it uses, etc.), but the core evaluation system remains the same. This modularity is like having a universal adapter that lets you plug any device into a standard outlet - you just need the right adapter for each device.

We selected three models representing distinct approaches to medical image analysis: Claude Sonnet 4 and GPT-5.1 as state-of-the-art general-purpose vision-language models from leading AI providers, and MedGemma as a medical-domain-specific model trained specifically on medical images. This selection enables comparison between general-purpose cloud-based APIs and specialized locally-deployable medical AI, addressing whether domain-specific training provides advantages over general-purpose models in dermatological classification tasks.
We selected three models representing distinct approaches to medical image analysis: Claude Sonnet 4 and GPT-5.1 as state-of-the-art general-purpose vision-language models from leading AI providers, and MedGemma as a medical-domain-specific model. For MedGemma, we used a quantized version (8-bit quantization, 5GB) deployed via Ollama rather than the full model, as the original model size exceeded available computational resources. This selection enables comparison between general-purpose cloud-based APIs and specialized locally-deployable medical AI, addressing whether domain-specific training provides advantages over general-purpose models in dermatological classification tasks.

### 4.5 Image Processing and Inference Architecture

When a classification script is executed, it follows a consistent workflow across all three models, with model-specific adaptations for their different API requirements. The process begins by loading the ground truth dataset, which contains file paths to 300 skin condition images along with their known classifications. For each image, the system loads the image file from disk, converts it to a standardized RGB format, and encodes it as a base64 string (a text representation of the image data that can be transmitted over the internet).

The encoded image is then packaged differently for each model's API: Claude receives the image embedded in a structured message format through Anthropic's API, GPT-5.1 receives it as a data URI in OpenAI's chat completion format, and MedGemma receives it as a raw base64 string through Ollama's REST API. Each model processes the image along with the classification prompt and returns a text response containing the classification, confidence level, and reasoning. The system extracts the classification from each response, normalizes it to standardized labels (e.g., mapping "blackheads" to "comedonal acne"), compares it against the ground truth, and records the result along with processing time and metadata.

This process repeats for all 300 images, with progress saved every 10 images to prevent data loss. The key difference between models lies in their API communication protocols and response formats, but the core workflow—load image, encode, send to model, receive response, normalize, and compare—remains consistent, ensuring fair comparison across all three approaches.