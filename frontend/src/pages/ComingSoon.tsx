export default function ComingSoon({ title, phase }: { title: string; phase: number }) {
  return (
    <section>
      <h1>{title}</h1>
      <p className="muted">Arrives in phase {phase} of the dashboard plan.</p>
    </section>
  );
}
