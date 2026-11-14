import { useQuery } from '@tanstack/react-query';
import { getJson, type AccountStatus, type StatusResponse } from '../api/client';

const ROLE_LABEL = { primary: 'Stocks (primary)', backup: 'Stocks (backup)', crypto: 'Crypto' };

function formatTime(iso: string) {
  return new Date(iso).toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
}

function AccountCard({ account }: { account: AccountStatus }) {
  if (!account.configured) {
    return <p className="muted">No Alpaca credentials set.</p>;
  }
  if (!account.ok) {
    return <p className="bad">Account call failed: {account.error}</p>;
  }
  return (
    <>
      <p>
        <span className={account.paper ? 'badge' : 'badge live'}>
          {account.paper ? 'Paper' : 'LIVE'}
        </span>
      </p>
      <dl className="kv">
        {Object.entries(account.balances ?? {}).map(([name, value]) => (
          <div key={name}>
            <dt>{name.replace('_', ' ')}</dt>
            <dd>{value.toLocaleString(undefined, { style: 'currency', currency: 'USD' })}</dd>
          </div>
        ))}
      </dl>
    </>
  );
}

export default function StatusPage() {
  const { data, error, isLoading, refetch, isFetching } = useQuery({
    queryKey: ['status'],
    queryFn: () => getJson<StatusResponse>('/status'),
  });

  return (
    <section>
      <div className="title-row">
        <h1>System status</h1>
        <button onClick={() => refetch()} disabled={isFetching}>
          {isFetching ? 'Checking…' : 'Refresh'}
        </button>
      </div>

      {isLoading && <p className="muted">Loading…</p>}
      {error && (
        <p className="bad">
          Could not reach the API. Is it running? (python -m uvicorn api.main:app --reload)
          <br />
          <small>{String(error)}</small>
        </p>
      )}

      {data && (
        <div className="grid">
          <article className="card">
            <h2>Data sources</h2>
            <ul className="plain">
              {data.sources.map((source) => (
                <li key={source.role}>
                  <span className={source.configured ? 'dot ok' : 'dot off'} />
                  {ROLE_LABEL[source.role]}: <code>{source.name || 'none'}</code>
                  {!source.configured && source.name && <span className="muted"> (not configured)</span>}
                </li>
              ))}
            </ul>
          </article>

          <article className="card">
            <h2>Alpaca account</h2>
            <AccountCard account={data.account} />
          </article>

          <article className="card wide">
            <h2>Stored data</h2>
            {data.store.length === 0 ? (
              <p className="muted">The DuckDB store is empty. Ingest some bars to see them here.</p>
            ) : (
              <table>
                <thead>
                  <tr><th>Symbol</th><th>Interval</th><th>Bars</th><th>First</th><th>Last</th></tr>
                </thead>
                <tbody>
                  {data.store.map((series) => (
                    <tr key={`${series.symbol}-${series.interval}`}>
                      <td>{series.symbol}</td>
                      <td>{series.interval}</td>
                      <td className="num">{series.n_bars.toLocaleString()}</td>
                      <td>{formatTime(series.first_timestamp)}</td>
                      <td>{formatTime(series.last_timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </article>

          <article className="card wide">
            <h2>Settings</h2>
            <p className="muted">Secrets are masked by the API.</p>
            <pre>{JSON.stringify(data.settings, null, 2)}</pre>
          </article>
        </div>
      )}
    </section>
  );
}
