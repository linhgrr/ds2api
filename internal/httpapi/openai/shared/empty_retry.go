package shared

import "strings"

const EmptyOutputRetrySuffix = "Previous reply had no visible output. Please regenerate the visible final answer or tool call now."
const StructuredOutputRetrySuffix = "Previous reply was rejected because it was not valid structured output. Regenerate exactly one valid JSON value that satisfies the required schema or JSON object constraints. Do not include markdown, prose, comments, or extra text."

func EmptyOutputRetryEnabled() bool {
	return true
}

func EmptyOutputRetryMaxAttempts() int {
	return 1
}

func StructuredOutputRetryMaxAttempts() int {
	return 2
}

func ClonePayloadWithEmptyOutputRetryPrompt(payload map[string]any) map[string]any {
	return ClonePayloadForEmptyOutputRetry(payload, 0)
}

// ClonePayloadForEmptyOutputRetry creates a retry payload with the suffix
// appended and, if parentMessageID > 0, sets parent_message_id so the
// retry is submitted as a proper follow-up turn in the same DeepSeek
// session rather than a disconnected root message.
func ClonePayloadForEmptyOutputRetry(payload map[string]any, parentMessageID int) map[string]any {
	clone := make(map[string]any, len(payload))
	for k, v := range payload {
		clone[k] = v
	}
	original, _ := payload["prompt"].(string)
	clone["prompt"] = AppendEmptyOutputRetrySuffix(original)
	if parentMessageID > 0 {
		clone["parent_message_id"] = parentMessageID
	}
	return clone
}

func AppendEmptyOutputRetrySuffix(prompt string) string {
	prompt = strings.TrimRight(prompt, "\r\n\t ")
	if prompt == "" {
		return EmptyOutputRetrySuffix
	}
	return prompt + "\n\n" + EmptyOutputRetrySuffix
}

func UsagePromptWithEmptyOutputRetry(originalPrompt string, retryAttempts int) string {
	if retryAttempts <= 0 {
		return originalPrompt
	}
	parts := make([]string, 0, retryAttempts+1)
	parts = append(parts, originalPrompt)
	next := originalPrompt
	for i := 0; i < retryAttempts; i++ {
		next = AppendEmptyOutputRetrySuffix(next)
		parts = append(parts, next)
	}
	return strings.Join(parts, "\n")
}

func ClonePayloadForStructuredOutputRetry(payload map[string]any, parentMessageID int, failureDetail string) map[string]any {
	clone := make(map[string]any, len(payload))
	for k, v := range payload {
		clone[k] = v
	}
	original, _ := payload["prompt"].(string)
	clone["prompt"] = AppendStructuredOutputRetrySuffix(original, failureDetail)
	if parentMessageID > 0 {
		clone["parent_message_id"] = parentMessageID
	}
	return clone
}

func AppendStructuredOutputRetrySuffix(prompt, failureDetail string) string {
	prompt = strings.TrimRight(prompt, "\r\n\t ")
	suffix := StructuredOutputRetrySuffix
	failureDetail = strings.TrimSpace(failureDetail)
	if failureDetail != "" {
		suffix += " Validation failure: " + failureDetail
	}
	if prompt == "" {
		return suffix
	}
	return prompt + "\n\n" + suffix
}
