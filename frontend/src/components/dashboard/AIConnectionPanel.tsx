"use client";

import { useCallback, useEffect, useId, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Field, Select, TextInput } from "@/components/ui/Field";
import { connectAI, currentConnection, disconnectAI, getProviders, signInServer } from "@/lib/api";
import type { AIConnection, AIProvider } from "@/lib/api";

export function AIConnectionPanel({onChange}: {onChange?: (connection: AIConnection | null) => void}) {
  const id = useId();
  const [{provider, model}, setSelection] = useState(() => ({
    provider: currentConnection()?.provider ?? "anthropic" as AIConnection["provider"],
    model: currentConnection()?.model ?? "",
  }));
  const [providers, setProviders] = useState<AIProvider[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [key, setKey] = useState("");
  const [token, setToken] = useState("");
  const [inputRate, setInputRate] = useState("");
  const [outputRate, setOutputRate] = useState("");
  const [connection, setConnection] = useState(currentConnection());
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const loadProviders = useCallback(async () => {
    setCatalogLoading(true);
    try {
      const data = await getProviders();
      setProviders(data.providers);
      setSelection(current => ({
        ...current,
        model: current.model || data.providers.find(p => p.id === current.provider)?.default_model || "",
      }));
      setMessage(data.offline ? "Server default: offline rules. Connect an AI to use it for this analysis." : "Use the server connection, or connect your own AI below.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not load models.");
    } finally {
      setCatalogLoading(false);
    }
  }, []);

  useEffect(() => { void loadProviders(); }, [loadProviders]);

  const selectedProvider = providers.find(p => p.id === provider);
  const models = selectedProvider?.models ?? [];
  const modelOptions = model && !models.some(option => option.id === model)
    ? [{id: model, label: model}, ...models]
    : models;

  async function connect() {
    setBusy(true);
    setMessage("");
    try {
      const result = await connectAI({provider, model, api_key: key,
        input_price_per_million: inputRate === "" ? undefined : Number(inputRate),
        output_price_per_million: outputRate === "" ? undefined : Number(outputRate)});
      setConnection(result);
      setKey("");
      onChange?.(result);
      setMessage("Connection verified. Available for one hour while the server stays running.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Connection failed."); }
    finally { setBusy(false); }
  }

  return <details className="rounded-xl border border-hairline bg-surface-2 p-4" open={Boolean(connection)}>
    <summary className="cursor-pointer text-sm font-medium text-ink">{connection ? `Connected: ${connection.provider} / ${connection.model}` : "Connect your AI"}</summary>
    <div className="mt-3 space-y-3">
      <p className="text-xs text-ink-3">Your key is held in server memory for this connection. It is excluded from saved runs and reports. Testing sends a small billed request to your provider.</p>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Provider" htmlFor={`${id}-provider`}>
          <Select id={`${id}-provider`} value={provider} disabled={busy || catalogLoading || !providers.length || Boolean(connection)} onChange={e => {
            const next = e.target.value as AIConnection["provider"];
            setSelection({provider: next, model: providers.find(p => p.id === next)?.default_model ?? ""});
          }}>
            <option value="anthropic">Claude (Anthropic)</option>
            <option value="openai">GPT (OpenAI)</option>
          </Select>
        </Field>
        <Field label="Model" htmlFor={`${id}-model`} hint="Availability depends on your API account. Connect and test checks access.">
          <Select id={`${id}-model`} value={model} aria-describedby={`${id}-model-hint`} disabled={busy || catalogLoading || !modelOptions.length || Boolean(connection)} onChange={e => setSelection({provider, model: e.target.value})}>
            {!model && <option value="">{catalogLoading ? "Loading models…" : "Models unavailable"}</option>}
            {modelOptions.map(option => <option key={option.id} value={option.id}>
              {option.label}{option.id === selectedProvider?.default_model ? " (server default)" : ""}
            </option>)}
          </Select>
        </Field>
      </div>
      {!catalogLoading && !providers.length && <Button type="button" onClick={() => void loadProviders()}>Retry loading models</Button>}
      {!connection && <>
        <Field label="API key" htmlFor={`${id}-key`}><TextInput id={`${id}-key`} type="password" autoComplete="off" value={key} onChange={e => setKey(e.target.value)} /></Field>
        <details><summary className="cursor-pointer text-xs text-ink-3">Cost estimates (needed for a dollar limit)</summary><div className="mt-2 grid gap-3 sm:grid-cols-2">
          <Field label="Input USD / million tokens" htmlFor={`${id}-input`}><TextInput id={`${id}-input`} type="number" min="0" step="any" value={inputRate} onChange={e => setInputRate(e.target.value)} /></Field>
          <Field label="Output USD / million tokens" htmlFor={`${id}-output`}><TextInput id={`${id}-output`} type="number" min="0" step="any" value={outputRate} onChange={e => setOutputRate(e.target.value)} /></Field>
        </div></details>
        <Button type="button" onClick={() => void connect()} disabled={busy || catalogLoading || !providers.length || !key || !model}>{busy ? "Testing connection…" : "Connect and test"}</Button>
      </>}
      {connection && <Button type="button" onClick={async () => {try {await disconnectAI(); setConnection(null); onChange?.(null); setMessage("Disconnected for future requests.");} catch (e) {setMessage(String(e));}}}>Disconnect</Button>}
      <p role="status" className="text-xs text-ink-2">{message}</p>
      <details><summary className="cursor-pointer text-xs text-ink-3">Server sign-in</summary><div className="mt-2 space-y-2"><Field label="Server access token" htmlFor={`${id}-server`}><TextInput id={`${id}-server`} type="password" autoComplete="off" value={token} onChange={e => setToken(e.target.value)} /></Field><Button type="button" onClick={async () => {try {await signInServer(token); setToken(""); await loadProviders();} catch (e) {setMessage(String(e));}}}>Sign in</Button></div></details>
    </div>
  </details>;
}
