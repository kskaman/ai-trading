import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ApiError, deleteJson, getJson, postJson, putJson,
  type ParameterInfo, type Setup, type SetupIn, type StrategyInfo,
  type UniversesResponse, type ValidationResult,
} from '../api/client';

// Inputs hold the text the user typed; an empty field means "use the default".
type FormValues = Record<string, string>;
type Step = 1 | 2 | 3 | 4;

const STEPS: { n: Step; label: string }[] = [
  { n: 1, label: 'Strategy' }, { n: 2, label: 'Symbols' }, { n: 3, label: 'Settings' }, { n: 4, label: 'Review' },
];
const PAIR_FIELDS = ['symbol_a', 'symbol_b'];

/** Show a stored value as input text (objects as JSON). */
function toText(value: unknown): string {
  if (value === null || value === undefined) return '';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

/** Split a typed symbol list on commas, spaces or new lines. */
function parseSymbols(text: string): string[] {
  return text.split(/[\s,;]+/).map((s) => s.trim()).filter(Boolean);
}

/**
 * Turn typed text into request params. Empty fields are left out so the
 * strategy's default applies; numbers that don't parse are sent as text so
 * the server explains the problem; bad JSON is caught here.
 */
function toParams(parameters: ParameterInfo[], values: FormValues) {
  const params: Record<string, unknown> = {};
  const localErrors: Record<string, string> = {};
  for (const p of parameters) {
    const text = (values[p.name] ?? '').trim();
    if (text === '') continue;
    if (p.type === 'integer' || p.type === 'number') {
      const n = Number(text);
      params[p.name] = Number.isNaN(n) ? text : n;
    } else if (p.type === 'object') {
      try { params[p.name] = JSON.parse(text); } catch { localErrors[p.name] = 'not valid JSON'; }
    } else {
      params[p.name] = text;
    }
  }
  return { params, localErrors };
}

/** Delay a fast-changing value so validation runs once typing pauses. */
function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return debounced;
}

/** The value a field will really use: what was typed, else its default. */
function effective(param: ParameterInfo | undefined, values: FormValues): number {
  if (!param) return 0;
  const typed = Number(values[param.name]);
  return values[param.name]?.trim() && !Number.isNaN(typed) ? typed : Number(param.default ?? 0);
}

function ParamField({ param, value, error, onChange }: {
  param: ParameterInfo; value: string; error?: string; onChange: (v: string) => void;
}) {
  const id = `param-${param.name}`;
  const numeric = param.type === 'integer' || param.type === 'number';
  const placeholder = param.default === null || param.default === undefined
    ? (param.required ? 'required' : 'leave empty for the default')
    : `default: ${toText(param.default)}`;
  return (
    <div className={error ? 'field has-error' : 'field'}>
      <label htmlFor={id}>
        {param.label || param.name}
        {param.required && <span className="req"> *</span>}
      </label>
      {param.type === 'object' ? (
        <textarea id={id} rows={2} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} />
      ) : (
        <input id={id} value={value} placeholder={placeholder}
          type={numeric ? 'number' : 'text'}
          min={param.minimum ?? undefined} max={param.maximum ?? undefined}
          step={param.step ?? (param.type === 'integer' ? 1 : 'any')}
          onChange={(e) => onChange(e.target.value)} />
      )}
      {error ? <small className="bad">{error}</small> : <small className="muted">{param.help}</small>}
    </div>
  );
}

export default function PaperTradePage() {
  const queryClient = useQueryClient();
  const strategies = useQuery({ queryKey: ['strategies'], queryFn: () => getJson<StrategyInfo[]>('/strategies') });
  const presets = useQuery({ queryKey: ['universes'], queryFn: () => getJson<UniversesResponse>('/universes') });
  const setups = useQuery({ queryKey: ['setups'], queryFn: () => getJson<Setup[]>('/setups') });

  const [step, setStep] = useState<Step>(1);
  const [strategyId, setStrategyId] = useState<string | null>(null);
  const [values, setValues] = useState<FormValues>({});
  const [symbolsText, setSymbolsText] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [name, setName] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedName, setSavedName] = useState<string | null>(null);

  const strategy = strategies.data?.find((s) => s.id === strategyId);
  const isPair = strategy?.kind === 'pair';
  const symbols = useMemo(() => (isPair ? [] : parseSymbols(symbolsText)), [isPair, symbolsText]);
  const { params, localErrors } = useMemo(
    () => toParams(strategy?.parameters ?? [], values), [strategy, values],
  );
  const request = useMemo(() => ({ params, symbols }), [params, symbols]);
  const debounced = useDebounced(request, 350);

  // Live validation: the server builds the real strategy with these values.
  const validation = useQuery({
    queryKey: ['validate', strategy?.id, debounced],
    queryFn: () => postJson<ValidationResult>(`/strategies/${strategy!.id}/validate`, debounced),
    enabled: !!strategy && Object.keys(localErrors).length === 0,
    staleTime: Infinity,
  });

  const resetForm = () => {
    setValues({}); setSymbolsText(''); setName(''); setEditingId(null);
    setSaveError(null); setShowAdvanced(false);
  };
  const chooseStrategy = (id: string) => { resetForm(); setStrategyId(id); setSavedName(null); setStep(2); };
  const setValue = (field: string, v: string) => setValues((old) => ({ ...old, [field]: v }));

  const save = useMutation({
    mutationFn: (body: SetupIn) =>
      editingId ? putJson<Setup>(`/setups/${editingId}`, body) : postJson<Setup>('/setups', body),
    onSuccess: (setup) => {
      queryClient.invalidateQueries({ queryKey: ['setups'] });
      setSavedName(setup.name); resetForm(); setStrategyId(null); setStep(1);
    },
    onError: (err) => {
      const detail = (err instanceof ApiError ? err.detail : null) as
        { errors?: Record<string, string>; error?: string } | string | null;
      const errors = typeof detail === 'object' ? detail?.errors : undefined;
      setSaveError(errors ? Object.values(errors).join('; ')
        : (typeof detail === 'object' && detail?.error) || err.message);
    },
  });
  const remove = useMutation({
    mutationFn: (id: string) => deleteJson(`/setups/${id}`),
    onSuccess: (_d, id) => {
      queryClient.invalidateQueries({ queryKey: ['setups'] });
      if (id === editingId) { resetForm(); setStrategyId(null); setStep(1); }
    },
  });

  const editSetup = (setup: Setup) => {
    setStrategyId(setup.strategy_id);
    setValues(Object.fromEntries(Object.entries(setup.params).map(([k, v]) => [k, toText(v)])));
    setSymbolsText(setup.symbols.join(', '));
    setName(setup.name); setEditingId(setup.id); setSaveError(null); setSavedName(null);
    setStep(3);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (strategies.isLoading) return <p className="muted">Loading strategies…</p>;
  if (strategies.error) return <p className="bad">Could not load strategies. Is the API running? {String(strategies.error)}</p>;

  const errors = { ...(validation.data?.errors ?? {}), ...localErrors };
  const upToDate = !!validation.data && debounced === request;
  const valid = upToDate && validation.data!.valid && Object.keys(localErrors).length === 0;
  const byName = (n: string) => strategy?.parameters.find((p) => p.name === n);
  const needed = isPair ? 2 : effective(byName('n_long'), values) + effective(byName('n_short'), values);
  const settings = strategy?.parameters.filter((p) => !PAIR_FIELDS.includes(p.name)) ?? [];
  const visible = settings.filter((p) => showAdvanced || !p.advanced || errors[p.name]);
  const hiddenCount = settings.filter((p) => p.advanced).length;
  const titleFor = (id: string) => strategies.data?.find((s) => s.id === id)?.title ?? id;
  const symbolProblem = isPair ? (errors.symbol_a || errors.symbol_b) : errors.symbols;
  const settingProblem = settings.some((p) => errors[p.name]) || validation.data?.error;

  return (
    <section>
      <div className="title-row">
        <h1>{editingId ? `Edit “${name}”` : 'Paper trade'}</h1>
        {strategy && <button onClick={() => { resetForm(); setStrategyId(null); setStep(1); }}>Start over</button>}
      </div>
      {savedName && <p className="ok">✓ Saved “{savedName}”. It appears in your saved setups below.</p>}

      <ol className="stepper">
        {STEPS.map((s) => (
          <li key={s.n} className={s.n === step ? 'current' : s.n < step ? 'done' : ''}>
            <button disabled={s.n > 1 && !strategy} onClick={() => setStep(s.n)}>
              <span className="num">{s.n}</span>{s.label}
            </button>
          </li>
        ))}
      </ol>

      {step === 1 && (
        <div className="choice-grid">
          {strategies.data!.map((s) => (
            <button key={s.id} className={s.id === strategyId ? 'choice active' : 'choice'} onClick={() => chooseStrategy(s.id)}>
              <strong>{s.title}</strong>
              <span className="muted">{s.explanation}</span>
              <span className="tag">{s.kind === 'pair' ? 'Needs 2 symbols that move together' : 'Needs a group of symbols'}</span>
            </button>
          ))}
        </div>
      )}

      {step === 2 && strategy && (
        <article className="card">
          <h2>{isPair ? 'Which two symbols?' : 'Which symbols should it choose from?'}</h2>
          {isPair ? (
            <>
              <p className="muted">Pick two that usually move together. Use two stocks, or two crypto pairs like BTC/USDT.</p>
              <div className="form-grid">
                {PAIR_FIELDS.map((f) => byName(f) && (
                  <ParamField key={f} param={byName(f)!} value={values[f] ?? ''} error={errors[f]} onChange={(v) => setValue(f, v)} />
                ))}
              </div>
              <p className="muted">Ideas:</p>
              <div className="chips">
                {presets.data?.pairs.map((p) => (
                  <button key={p.label} className="chip" onClick={() => setValues((old) => ({ ...old, symbol_a: p.a, symbol_b: p.b }))}>
                    {p.label}
                  </button>
                ))}
              </div>
            </>
          ) : (
            <>
              <p className="muted">
                The strategy ranks these against each other, then buys the best and bets against the worst.
                Type symbols separated by commas, or start from a ready-made list and edit it.
              </p>
              <div className="chips">
                {presets.data?.universes.map((u) => (
                  <button key={u.id} className="chip" title={u.description} onClick={() => setSymbolsText(u.symbols.join(', '))}>
                    {u.label} ({u.symbols.length})
                  </button>
                ))}
              </div>
              <textarea className="symbols-box" rows={4} value={symbolsText} placeholder="AAPL, MSFT, NVDA, …"
                onChange={(e) => setSymbolsText(e.target.value)} />
              <p className={symbols.length >= needed ? 'muted' : 'warn'}>
                {symbols.length} symbols · needs at least {needed} (number to buy + number to bet against, set in the next step)
              </p>
            </>
          )}
          {symbolProblem && !isPair && <p className="bad">{symbolProblem}</p>}
          <div className="nav-row">
            <button onClick={() => setStep(1)}>Back</button>
            <button className="primary" onClick={() => setStep(3)}>Next: settings</button>
          </div>
        </article>
      )}

      {step === 3 && strategy && (
        <article className="card">
          <h2>Settings for {strategy.title}</h2>
          <p className="muted">The defaults are a sensible starting point. Leave a box empty to use its default.</p>
          <div className="form-grid">
            {visible.map((p) => (
              <ParamField key={p.name} param={p} value={values[p.name] ?? ''} error={errors[p.name]} onChange={(v) => setValue(p.name, v)} />
            ))}
          </div>
          {hiddenCount > 0 && (
            <button className="link" onClick={() => setShowAdvanced((s) => !s)}>
              {showAdvanced ? 'Hide advanced settings' : `Show ${hiddenCount} advanced setting${hiddenCount > 1 ? 's' : ''}`}
            </button>
          )}
          <p className="validation">
            {validation.isFetching && <span className="muted">Checking…</span>}
            {!validation.isFetching && validation.data?.error && <span className="bad">{validation.data.error}</span>}
          </p>
          <div className="nav-row">
            <button onClick={() => setStep(2)}>Back</button>
            <button className="primary" onClick={() => setStep(4)}>Next: review</button>
          </div>
        </article>
      )}

      {step === 4 && strategy && (
        <article className="card">
          <h2>Review</h2>
          <dl className="review">
            <dt>Strategy</dt><dd>{strategy.title}</dd>
            <dt>Symbols</dt>
            <dd>
              {isPair ? `${values.symbol_a?.toUpperCase() || '—'} and ${values.symbol_b?.toUpperCase() || '—'}` : `${symbols.length} symbols: ${symbols.slice(0, 12).join(', ')}${symbols.length > 12 ? ', …' : ''}`}
              {symbolProblem && <span className="bad"> ({symbolProblem}) <button className="link" onClick={() => setStep(2)}>fix</button></span>}
            </dd>
            <dt>Settings</dt>
            <dd>
              {settings.map((p) => `${p.label || p.name}: ${values[p.name]?.trim() || toText(p.default)}`).join(' · ')}
              {settingProblem && <span className="bad"> (needs attention) <button className="link" onClick={() => setStep(3)}>fix</button></span>}
            </dd>
          </dl>
          {validation.isFetching ? <p className="muted">Checking…</p> : valid && <p className="ok">✓ Ready to go</p>}

          <div className="save-row">
            <input aria-label="Name" placeholder="Give it a name, e.g. Momentum on Dow 30" value={name} onChange={(e) => setName(e.target.value)} />
            <button className="primary" disabled={!valid || !name.trim() || save.isPending}
              onClick={() => save.mutate({ name, strategy_id: strategy.id, params, symbols })}>
              {editingId ? 'Save changes' : 'Save setup'}
            </button>
            <button disabled title="Paper trading arrives in the next build step">Start paper trading</button>
          </div>
          {saveError && <p className="bad">{saveError}</p>}
          <div className="nav-row"><button onClick={() => setStep(3)}>Back</button></div>
        </article>
      )}

      <article className="card setups">
        <h2>Saved setups</h2>
        {setups.data?.length ? (
          <table>
            <thead><tr><th>Name</th><th>Strategy</th><th>Symbols</th><th>Updated</th><th /></tr></thead>
            <tbody>
              {setups.data.map((setup) => (
                <tr key={setup.id} className={setup.id === editingId ? 'editing' : undefined}>
                  <td>{setup.name}</td>
                  <td>{titleFor(setup.strategy_id)}</td>
                  <td className="params-cell">
                    {setup.symbols.length
                      ? `${setup.symbols.length}: ${setup.symbols.slice(0, 6).join(', ')}${setup.symbols.length > 6 ? ', …' : ''}`
                      : `${toText(setup.params.symbol_a)} / ${toText(setup.params.symbol_b)}`}
                  </td>
                  <td>{new Date(setup.updated_at).toLocaleString()}</td>
                  <td className="actions">
                    <button onClick={() => editSetup(setup)}>Edit</button>
                    <button onClick={() => { if (confirm(`Delete “${setup.name}”?`)) remove.mutate(setup.id); }}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">No saved setups yet. Start a strategy above and save it.</p>
        )}
      </article>
    </section>
  );
}
