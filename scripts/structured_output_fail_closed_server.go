package main

import (
	"context"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/go-chi/chi/v5"

	"ds2api/internal/auth"
	dsclient "ds2api/internal/deepseek/client"
	"ds2api/internal/httpapi/openai/chat"
	"ds2api/internal/httpapi/openai/responses"
	"ds2api/internal/httpapi/openai/shared"
)

type mockConfig struct{}

func (mockConfig) ModelAliases() map[string]string     { return nil }
func (mockConfig) ToolcallMode() string                { return "" }
func (mockConfig) ToolcallEarlyEmitConfidence() string { return "" }
func (mockConfig) ResponsesStoreTTLSeconds() int       { return 0 }
func (mockConfig) EmbeddingsProvider() string          { return "deterministic" }
func (mockConfig) AutoDeleteMode() string              { return "none" }
func (mockConfig) AutoDeleteSessions() bool            { return false }
func (mockConfig) CurrentInputFileEnabled() bool       { return false }
func (mockConfig) CurrentInputFileMinChars() int       { return 0 }
func (mockConfig) ThinkingInjectionEnabled() bool      { return false }
func (mockConfig) ThinkingInjectionPrompt() string     { return "" }

type allowAllAuth struct{}

func (allowAllAuth) Determine(r *http.Request) (*auth.RequestAuth, error) {
	return authFromRequest(r), nil
}

func (allowAllAuth) DetermineCaller(r *http.Request) (*auth.RequestAuth, error) {
	return authFromRequest(r), nil
}

func (allowAllAuth) Release(_ *auth.RequestAuth) {}

func authFromRequest(r *http.Request) *auth.RequestAuth {
	token := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer"))
	if token == "" {
		token = "test-token"
	}
	return &auth.RequestAuth{
		UseConfigToken: false,
		DeepSeekToken:  token,
		CallerID:       "caller:structured-fail-closed",
		TriedAccounts:  map[string]bool{},
	}
}

type failClosedDS struct {
	callSeq atomic.Int64
}

func (f *failClosedDS) CreateSession(_ context.Context, _ *auth.RequestAuth, _ int) (string, error) {
	return "session-fail-closed", nil
}

func (f *failClosedDS) GetPow(_ context.Context, _ *auth.RequestAuth, _ int) (string, error) {
	return "pow-bypass", nil
}

func (f *failClosedDS) UploadFile(_ context.Context, _ *auth.RequestAuth, _ dsclient.UploadFileRequest, _ int) (*dsclient.UploadFileResult, error) {
	return &dsclient.UploadFileResult{ID: "file-mock-1", Filename: "mock.txt", Bytes: 1, Status: "uploaded"}, nil
}

func (f *failClosedDS) CallCompletion(_ context.Context, _ *auth.RequestAuth, _ map[string]any, _ string, _ int) (*http.Response, error) {
	n := f.callSeq.Add(1)
	var payload string
	switch (n - 1) % 3 {
	case 0:
		payload = `{"score":"a"}`
	case 1:
		payload = `{"score":"b"}`
	default:
		payload = `{"score":"c"}`
	}
	responseID := 1000 + n
	return sseResponse(
		`data: {"response_message_id":`+strconv.FormatInt(responseID, 10)+`,"p":"response/content","v":"`+escapeForJSON(payload)+`"}`,
		`data: [DONE]`,
	), nil
}

func (f *failClosedDS) DeleteSessionForToken(_ context.Context, _ string, _ string) (*dsclient.DeleteSessionResult, error) {
	return &dsclient.DeleteSessionResult{Success: true}, nil
}

func (f *failClosedDS) DeleteAllSessionsForToken(_ context.Context, _ string) error {
	return nil
}

func sseResponse(lines ...string) *http.Response {
	body := strings.Join(lines, "\n")
	if !strings.HasSuffix(body, "\n") {
		body += "\n"
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func escapeForJSON(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	return s
}

func main() {
	port := strings.TrimSpace(os.Getenv("PORT"))
	if port == "" {
		port = "5011"
	}

	store := mockConfig{}
	authResolver := allowAllAuth{}
	ds := &failClosedDS{}

	chatHandler := &chat.Handler{Store: store, Auth: authResolver, DS: ds, ChatHistory: nil}
	responsesHandler := &responses.Handler{Store: store, Auth: authResolver, DS: ds, ChatHistory: nil}
	modelsHandler := &shared.ModelsHandler{Store: store}

	r := chi.NewRouter()
	r.Get("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		shared.WriteJSON(w, http.StatusOK, map[string]any{"ok": true})
	})
	r.Get("/v1/models", modelsHandler.ListModels)
	r.Get("/v1/models/{model_id}", modelsHandler.GetModel)
	r.Post("/v1/chat/completions", chatHandler.ChatCompletions)
	r.Post("/v1/responses", responsesHandler.Responses)

	addr := ":" + port
	log.Printf("structured-output fail-closed mock provider listening on %s", addr)
	if err := http.ListenAndServe(addr, r); err != nil {
		log.Fatal(err)
	}
}
