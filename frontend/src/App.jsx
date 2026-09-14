import { useMemo, useState } from 'react'
import { ArrowUpRight, CarFront, Check, Gauge, RotateCcw, Sparkles } from 'lucide-react'

// The model returns prices in Indian rupee lakh; display them in Sri Lankan rupee lakh.
const INR_TO_LKR_RATE = 3.75

const initialForm = {
  year: 2015,
  kilometers_driven: 41000,
  fuel_type: 'Diesel',
  transmission: 'Manual',
  owner_type: 'First',
  mileage: 19.67,
  engine: 1582,
  power: 126.2,
  seats: 5,
  location: 'Pune',
}

const fields = [
  { name: 'year', label: 'Model year', type: 'number', min: 1980, max: 2030, step: 1 },
  { name: 'kilometers_driven', label: 'Distance driven', suffix: 'km', type: 'number', min: 0, step: 1000 },
  { name: 'mileage', label: 'Mileage', suffix: 'kmpl', type: 'number', min: 0, step: 0.01 },
  { name: 'engine', label: 'Engine size', suffix: 'cc', type: 'number', min: 1, step: 1 },
  { name: 'power', label: 'Power', suffix: 'bhp', type: 'number', min: 0, step: 0.1 },
  { name: 'seats', label: 'Seats', type: 'number', min: 1, step: 1 },
]

function App() {
  const [form, setForm] = useState(initialForm)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const vehicleAge = useMemo(() => Math.max(0, 2026 - Number(form.year)), [form.year])
  const usage = useMemo(() => Math.round(Number(form.kilometers_driven) / Math.max(vehicleAge, 1)), [form.kilometers_driven, vehicleAge])
  const priceLkrLakh = result ? result.predicted_price_lakh * INR_TO_LKR_RATE : null

  function update(name, value) {
    setForm((current) => ({ ...current, [name]: value }))
    setResult(null)
    setError('')
  }

  async function submit(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          year: Number(form.year),
          kilometers_driven: Number(form.kilometers_driven),
          mileage: Number(form.mileage),
          engine: Number(form.engine),
          power: Number(form.power),
          seats: Number(form.seats),
        }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'The appraisal could not be completed.')
      setResult(data)
    } catch (requestError) {
      setError(requestError.message || 'Unable to reach the prediction service.')
    } finally {
      setLoading(false)
    }
  }

  function reset() {
    setForm(initialForm)
    setResult(null)
    setError('')
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="DriveValue home">
          <span className="brand-mark"><CarFront size={21} strokeWidth={2.4} /></span>
          <span>DRIVE<span>VALUE</span></span>
        </a>
        <div className="status"><span className="status-dot" /> Model online <span className="status-divider" /> GradientBoosting</div>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow"><Sparkles size={15} /> Dealership intelligence</p>
          <h1>Know what it’s<br /><em>really worth.</em></h1>
          <p className="intro-copy">A sharper starting point for every used-car conversation. Enter the vehicle details and get a model-backed resale estimate in seconds.</p>
        </div>
        <div className="intro-stamp"><span>01</span><small>APPRAISAL<br />DESK</small></div>
      </section>

      <section className="workspace">
        <form className="form-panel" onSubmit={submit}>
          <div className="panel-heading"><div><span className="section-number">01</span><h2>Vehicle profile</h2></div><span className="required">All fields required</span></div>
          <div className="field-grid">
            {fields.map((field) => (
              <label className="field" key={field.name}>
                <span>{field.label}</span>
                <div className="input-wrap"><input required {...field} value={form[field.name]} onChange={(event) => update(field.name, event.target.value)} />{field.suffix && <small>{field.suffix}</small>}</div>
              </label>
            ))}
          </div>

          <div className="panel-heading second-heading"><div><span className="section-number">02</span><h2>Configuration</h2></div></div>
          <div className="choice-grid">
            <SelectField label="Fuel type" name="fuel_type" value={form.fuel_type} options={['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric']} onChange={update} />
            <SelectField label="Transmission" name="transmission" value={form.transmission} options={['Manual', 'Automatic']} onChange={update} />
            <SelectField label="Ownership" name="owner_type" value={form.owner_type} options={['First', 'Second', 'Third', 'Fourth & Above']} onChange={update} />
            <label className="field"><span>City</span><input required value={form.location} onChange={(event) => update('location', event.target.value)} /></label>
          </div>
          {error && <p className="error-message">{error}</p>}
          <div className="form-actions"><button className="primary-button" type="submit" disabled={loading}>{loading ? 'Calculating...' : 'Calculate market value'} <ArrowUpRight size={18} /></button><button className="reset-button" type="button" onClick={reset}><RotateCcw size={15} /> Reset</button></div>
        </form>

        <aside className="insight-panel">
          <div className="insight-top"><span className="section-number light">03</span><span className="live-label">LIVE READOUT</span></div>
          <div className="readout-copy"><p>Vehicle signal</p><h2>{result ? 'Valuation ready' : 'Build your signal'}</h2><span>{result ? 'Based on current model inputs' : 'Your estimate will appear here'}</span></div>
          <div className="signal-card"><div className="signal-label"><Gauge size={17} /> Usage profile</div><div className="signal-value">{usage.toLocaleString()} <small>km / year</small></div><div className="signal-track"><span style={{ width: `${Math.min(100, Math.max(8, usage / 1200))}%` }} /></div><p>{vehicleAge} year{vehicleAge === 1 ? '' : 's'} old · {form.transmission} · {form.fuel_type}</p></div>
          <div className={`price-card ${result ? 'is-ready' : ''}`}><span>Estimated resale value</span>{result ? <strong>LKR {priceLkrLakh.toFixed(2)} <small>Lakh</small></strong> : <strong className="placeholder">LKR — <small>Lakh</small></strong>}<div className="price-foot"><Check size={14} /> {result ? 'Model confidence signal received' : 'Complete the profile to calculate'}</div><p className="currency-note">Converted from INR at 1 INR = {INR_TO_LKR_RATE} LKR</p></div>
          <p className="disclaimer">This estimate is a data-informed starting point, not a guaranteed transaction price. Final value depends on inspection and local market conditions.</p>
        </aside>
      </section>
      <footer><span>DRIVEVALUE / 2026</span><span>USED CAR PRICE INTELLIGENCE</span></footer>
    </main>
  )
}

function SelectField({ label, name, value, options, onChange }) {
  return <label className="field"><span>{label}</span><select value={value} onChange={(event) => onChange(name, event.target.value)}>{options.map((option) => <option key={option}>{option}</option>)}</select></label>
}

export default App
