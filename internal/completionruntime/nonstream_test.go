package completionruntime

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"ds2api/internal/account"
	"ds2api/internal/auth"
	"ds2api/internal/config"
	dsclient "ds2api/internal/deepseek/client"
	"ds2api/internal/httpapi/openai/shared"
	"ds2api/internal/promptcompat"
)

type fakeDeepSeekCaller struct {
	createSessions int
	responses      []*http.Response
	payloads       []map[string]any
	uploads        []dsclient.UploadFileRequest
	accountIDs     []string
}

type currentInputRuntimeConfig struct{}

func (currentInputRuntimeConfig) CurrentInputFileEnabled() bool { return true }
func (currentInputRuntimeConfig) CurrentInputFileMinChars() int { return 0 }

func (f *fakeDeepSeekCaller) GetPow(context.Context, *auth.RequestAuth, int) (string, error) {
	return "pow", nil
}

func (f *fakeDeepSeekCaller) UploadFile(_ context.Context, _ *auth.RequestAuth, req dsclient.UploadFileRequest, _ int) (*dsclient.UploadFileResult, error) {
	f.uploads = append(f.uploads, req)
	return &dsclient.UploadFileResult{ID: "file-runtime-1"}, nil
}

func (f *fakeDeepSeekCaller) CallCompletion(_ context.Context, a *auth.RequestAuth, payload map[string]any, _ string, _ int) (*http.Response, error) {
	f.payloads = append(f.payloads, payload)
	if a != nil {
		f.accountIDs = append(f.accountIDs, a.AccountID)
	}
	if len(f.responses) == 0 {
		return sseHTTPResponse(http.StatusOK, `data: {"p":"response/content","v":"fallback"}`), nil
	}
	resp := f.responses[0]
	f.responses = f.responses[1:]
	return resp, nil
}

func (f *fakeDeepSeekCaller) CreateSession(_ context.Context, a *auth.RequestAuth, _ int) (string, error) {
	f.createSessions++
	if a != nil && strings.TrimSpace(a.AccountID) != "" {
		return "session-" + a.AccountID, nil
	}
	return "session-1", nil
}

func TestExecuteNonStreamWithRetryBuildsCanonicalTurn(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{sseHTTPResponse(
		http.StatusOK,
		`data: {"response_message_id":42,"p":"response/content","v":"<tool_calls><invoke name=\"Write\"><parameter name=\"content\">{\"x\":1}</parameter></invoke></tool_calls>"}`,
	)}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
		ToolNames:       []string{"Write"},
		ToolsRaw: []any{map[string]any{
			"name": "Write",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"content": map[string]any{"type": "string"},
				},
			},
		}},
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{}, stdReq, Options{})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if result.SessionID != "session-1" {
		t.Fatalf("session mismatch: %q", result.SessionID)
	}
	if got := result.Turn.ResponseMessageID; got != 42 {
		t.Fatalf("response message id mismatch: %d", got)
	}
	if len(result.Turn.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(result.Turn.ToolCalls))
	}
	if _, ok := result.Turn.ToolCalls[0].Input["content"].(string); !ok {
		t.Fatalf("expected schema-normalized string argument, got %#v", result.Turn.ToolCalls[0].Input["content"])
	}
	if result.Turn.Usage.InputTokens == 0 || result.Turn.Usage.TotalTokens == 0 {
		t.Fatalf("expected usage to be populated, got %#v", result.Turn.Usage)
	}
}

func TestExecuteNonStreamWithRetryUsesParentMessageForEmptyRetry(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":77,"p":"response/status","v":"FINISHED"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":78,"p":"response/content","v":"ok"}`),
	}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{}, stdReq, Options{RetryEnabled: true})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if result.Attempts != 1 {
		t.Fatalf("expected one retry, got %d", result.Attempts)
	}
	if len(ds.payloads) != 2 {
		t.Fatalf("expected two completion calls, got %d", len(ds.payloads))
	}
	if got := ds.payloads[1]["parent_message_id"]; got != 77 {
		t.Fatalf("retry parent_message_id mismatch: %#v", got)
	}
	if result.Turn.Text != "ok" {
		t.Fatalf("retry text mismatch: %q", result.Turn.Text)
	}
}

func TestExecuteNonStreamWithRetryConvertsReferenceMarkers(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{sseHTTPResponse(
		http.StatusOK,
		`data: {"p":"response/content","v":"答案[reference:0]。","citation":{"cite_index":0,"url":"https://example.com/ref"}}`,
	)}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash-search",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
		Search:          true,
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{}, stdReq, Options{})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	want := "答案[0](https://example.com/ref)。"
	if result.Turn.Text != want {
		t.Fatalf("text mismatch: got %q want %q", result.Turn.Text, want)
	}
}

func TestStartCompletionAppliesCurrentInputFileGlobally(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{sseHTTPResponse(http.StatusOK, `data: {"p":"response/content","v":"ok"}`)}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test_adapter",
		RequestedModel:  "deepseek-v4-flash",
		ResolvedModel:   "deepseek-v4-flash",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "first user turn",
		FinalPrompt:     "first user turn",
		Messages: []any{
			map[string]any{"role": "user", "content": "first user turn"},
		},
	}

	start, outErr := StartCompletion(context.Background(), ds, &auth.RequestAuth{DeepSeekToken: "token"}, stdReq, Options{
		CurrentInputFile: currentInputRuntimeConfig{},
	})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if len(ds.uploads) != 1 {
		t.Fatalf("expected current input upload, got %d", len(ds.uploads))
	}
	if got := ds.uploads[0].Filename; got != "DS2API_HISTORY.txt" {
		t.Fatalf("upload filename=%q want DS2API_HISTORY.txt", got)
	}
	if len(ds.payloads) != 1 {
		t.Fatalf("expected one completion payload, got %d", len(ds.payloads))
	}
	refIDs, _ := ds.payloads[0]["ref_file_ids"].([]any)
	if len(refIDs) != 1 || refIDs[0] != "file-runtime-1" {
		t.Fatalf("expected uploaded file id in ref_file_ids, got %#v", ds.payloads[0]["ref_file_ids"])
	}
	prompt, _ := ds.payloads[0]["prompt"].(string)
	if !strings.Contains(prompt, "Continue from the latest state in the attached DS2API_HISTORY.txt context.") {
		t.Fatalf("expected continuation prompt, got %q", prompt)
	}
	if !start.Request.CurrentInputFileApplied || !strings.Contains(start.Request.PromptTokenText, "# DS2API_HISTORY.txt") {
		t.Fatalf("expected prepared request to carry current input file state, got %#v", start.Request)
	}
}

func TestStartCompletionSkipsCurrentInputFileForStructuredOutput(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{sseHTTPResponse(http.StatusOK, `data: {"p":"response/content","v":"{\"ok\":true}"}`)}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test_adapter",
		RequestedModel:  "deepseek-v4-flash",
		ResolvedModel:   "deepseek-v4-flash",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "first user turn",
		FinalPrompt:     "first user turn",
		Messages: []any{
			map[string]any{"role": "user", "content": "first user turn"},
		},
		StructuredOutput: &promptcompat.StructuredOutputSpec{Mode: "json_object"},
	}

	start, outErr := StartCompletion(context.Background(), ds, &auth.RequestAuth{DeepSeekToken: "token"}, stdReq, Options{
		CurrentInputFile: currentInputRuntimeConfig{},
	})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if len(ds.uploads) != 0 {
		t.Fatalf("expected current input upload to be skipped, got %d", len(ds.uploads))
	}
	if start.Request.CurrentInputFileApplied {
		t.Fatalf("expected current input file to remain disabled for structured output")
	}
}

func TestExecuteNonStreamWithRetryUsesParentMessageAndPromptCorrectionForStructuredOutput(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":91,"p":"response/content","v":"{\"score\":\"9\"}"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":92,"p":"response/content","v":"{\"score\":9}"}`),
	}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
		StructuredOutput: &promptcompat.StructuredOutputSpec{
			Mode: "json_schema",
			Schema: map[string]any{
				"type": "object",
				"required": []any{
					"score",
				},
				"properties": map[string]any{
					"score": map[string]any{"type": "integer"},
				},
			},
		},
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{}, stdReq, Options{RetryEnabled: true})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if ds.createSessions != 1 {
		t.Fatalf("expected one session, got %d", ds.createSessions)
	}
	if len(ds.payloads) != 2 {
		t.Fatalf("expected two completion calls, got %d", len(ds.payloads))
	}
	if got := ds.payloads[1]["parent_message_id"]; got != 91 {
		t.Fatalf("retry parent_message_id mismatch: %#v", got)
	}
	prompt, _ := ds.payloads[1]["prompt"].(string)
	if !strings.Contains(prompt, shared.StructuredOutputRetrySuffix) {
		t.Fatalf("expected structured output retry prompt suffix, got %q", prompt)
	}
	if result.Turn.Text != `{"score":9}` {
		t.Fatalf("expected canonical structured output, got %q", result.Turn.Text)
	}
}

func TestExecuteNonStreamWithRetryReturns422AfterStructuredOutputRetryLimit(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":101,"p":"response/content","v":"{\"score\":\"a\"}"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":102,"p":"response/content","v":"{\"score\":\"b\"}"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":103,"p":"response/content","v":"{\"score\":\"c\"}"}`),
	}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
		StructuredOutput: &promptcompat.StructuredOutputSpec{
			Mode: "json_schema",
			Schema: map[string]any{
				"type": "object",
				"required": []any{
					"score",
				},
				"properties": map[string]any{
					"score": map[string]any{"type": "integer"},
				},
			},
		},
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{}, stdReq, Options{RetryEnabled: true})
	if outErr == nil {
		t.Fatal("expected structured output validation error")
	}
	if outErr.Status != http.StatusUnprocessableEntity || outErr.Code != "structured_output_validation_failed" {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if len(ds.payloads) != 3 {
		t.Fatalf("expected initial call plus two retries, got %d", len(ds.payloads))
	}
	if result.Turn.ResponseMessageID != 103 {
		t.Fatalf("expected final failed turn to be returned, got %#v", result.Turn.ResponseMessageID)
	}
}

func TestExecuteNonStreamWithRetryManagedFailoverUsesRoundRobinAccounts(t *testing.T) {
	a := managedTestAuth(t)
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":201,"p":"response/status","v":"FINISHED"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":202,"p":"response/status","v":"FINISHED"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":203,"p":"response/content","v":"ok"}`),
	}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{RetryEnabled: true})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if result.Attempts != 2 {
		t.Fatalf("expected two managed-account retries, got %d", result.Attempts)
	}
	if ds.createSessions != 3 {
		t.Fatalf("expected one fresh session per account attempt, got %d", ds.createSessions)
	}
	wantAccounts := []string{"alpha@example.com", "beta@example.com", "gamma@example.com"}
	if strings.Join(ds.accountIDs, ",") != strings.Join(wantAccounts, ",") {
		t.Fatalf("account sequence mismatch: got %v want %v", ds.accountIDs, wantAccounts)
	}
	if len(ds.payloads) != 3 {
		t.Fatalf("expected three completion calls, got %d", len(ds.payloads))
	}
	if got := ds.payloads[1]["parent_message_id"]; got != nil {
		t.Fatalf("expected full failover replay instead of same-session retry, got parent_message_id=%#v", got)
	}
	if result.Turn.Text != "ok" {
		t.Fatalf("expected final success text, got %q", result.Turn.Text)
	}
}

func TestExecuteNonStreamWithRetryManagedFailoverExhaustsRetryBudget(t *testing.T) {
	a := managedTestAuth(t)
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":301,"p":"response/content","v":"{\"score\":\"a\"}"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":302,"p":"response/content","v":"{\"score\":\"b\"}"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":303,"p":"response/content","v":"{\"score\":\"c\"}"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":304,"p":"response/content","v":"{\"score\":\"d\"}"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":305,"p":"response/content","v":"{\"score\":\"e\"}"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":306,"p":"response/content","v":"{\"score\":\"f\"}"}`),
	}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
		StructuredOutput: &promptcompat.StructuredOutputSpec{
			Mode: "json_schema",
			Schema: map[string]any{
				"type": "object",
				"required": []any{
					"score",
				},
				"properties": map[string]any{
					"score": map[string]any{"type": "integer"},
				},
			},
		},
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{RetryEnabled: true})
	if outErr == nil {
		t.Fatal("expected managed-account retry exhaustion error")
	}
	if outErr.Status != http.StatusUnprocessableEntity || outErr.Code != "structured_output_validation_failed" {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if !strings.Contains(outErr.Message, "All managed-account retries failed after 6 attempts") {
		t.Fatalf("expected exhaustion detail in error message, got %q", outErr.Message)
	}
	if result.Attempts != 5 {
		t.Fatalf("expected five retries after the initial attempt, got %d", result.Attempts)
	}
	if ds.createSessions != 6 {
		t.Fatalf("expected six fresh sessions, got %d", ds.createSessions)
	}
	wantAccounts := []string{
		"alpha@example.com",
		"beta@example.com",
		"gamma@example.com",
		"alpha@example.com",
		"beta@example.com",
		"gamma@example.com",
	}
	if strings.Join(ds.accountIDs, ",") != strings.Join(wantAccounts, ",") {
		t.Fatalf("account sequence mismatch: got %v want %v", ds.accountIDs, wantAccounts)
	}
}

func managedTestAuth(t *testing.T) *auth.RequestAuth {
	t.Helper()
	const rawConfig = `{"accounts":[{"email":"alpha@example.com","password":"pw1"},{"email":"beta@example.com","password":"pw2"},{"email":"gamma@example.com","password":"pw3"}],"keys":["proxypal-local"]}`
	prev := os.Getenv("DS2API_CONFIG_JSON")
	if err := os.Setenv("DS2API_CONFIG_JSON", rawConfig); err != nil {
		t.Fatalf("set env: %v", err)
	}
	t.Cleanup(func() {
		if prev == "" {
			_ = os.Unsetenv("DS2API_CONFIG_JSON")
			return
		}
		_ = os.Setenv("DS2API_CONFIG_JSON", prev)
	})

	store, err := config.LoadStoreWithError()
	if err != nil {
		t.Fatalf("load store: %v", err)
	}
	resolver := auth.NewResolver(store, account.NewPool(store), func(_ context.Context, acc config.Account) (string, error) {
		return "token-for-" + acc.Email, nil
	})
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer proxypal-local")
	a, err := resolver.Determine(req)
	if err != nil {
		t.Fatalf("determine auth: %v", err)
	}
	t.Cleanup(func() { resolver.Release(a) })
	return a
}

func sseHTTPResponse(status int, lines ...string) *http.Response {
	body := strings.Join(lines, "\n")
	if !strings.HasSuffix(body, "\n") {
		body += "\n"
	}
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
