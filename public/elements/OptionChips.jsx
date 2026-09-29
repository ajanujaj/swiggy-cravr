import { useState } from 'react';

function groupOptions(options) {
  const groups = [];
  for (const o of options) {
    const last = groups[groups.length - 1];
    if (!last || last.name !== (o.group || '')) {
      groups.push({ name: o.group || '', required: !!o.required, items: [o] });
    } else {
      last.items.push(o);
    }
  }
  return groups;
}

export default function OptionChips() {
  const options = props.options || [];
  const multi = !!props.multi;
  // Quantity steppers only show up for a dish multi-select (never for
  // category picks or the addon-selection interrupt, which reuse this same
  // component in `multi` mode without setting `quantities`) -- see
  // UI/graph_bridge.py's allow_quantity / UI/components.py's
  // send_multiselect_message(allow_quantity=...).
  const allowQty = multi && !!props.quantities;
  const [picked, setPicked] = useState(null);
  const [chosen, setChosen] = useState([]);
  const [qty, setQty] = useState({});
  const locked = !!props.locked || picked !== null;
  const selected = props.selected ?? picked;
  const selectedList = multi ? (props.selected || chosen) : [];

  const pick = (o) => {
    if (locked) return;
    setPicked(o.value);
    callAction({ name: 'pick', payload: { value: o.value, label: o.label } });
  };
  const toggle = (o) => {
    if (locked) return;
    setChosen((c) => {
      if (c.includes(o.value)) {
        setQty((q) => {
          const next = { ...q };
          delete next[o.value];
          return next;
        });
        return c.filter((v) => v !== o.value);
      }
      setQty((q) => ({ ...q, [o.value]: q[o.value] || 1 }));
      return [...c, o.value];
    });
  };
  const bumpQty = (value, delta) => {
    if (locked) return;
    setQty((q) => ({ ...q, [value]: Math.max(1, Math.min(20, (q[value] || 1) + delta)) }));
  };
  const submit = () => {
    if (locked) return;
    setPicked(chosen);
    const values = allowQty
      ? chosen.map((v) => (qty[v] > 1 ? `${v}:${qty[v]}` : v))
      : chosen;
    callAction({ name: 'multiselect_submit', payload: { values } });
  };

  const chip = (o) => {
    const on = multi ? selectedList.includes(o.value) : selected === o.value;
    const cls = 'cravr-chip' + (on ? (multi ? ' is-on' : ' is-picked') : '');
    return (
      <span key={o.value} className="cravr-chip-row">
        <button className={cls} disabled={locked} onClick={() => (multi ? toggle(o) : pick(o))}>
          {multi && on ? <span className="cravr-tick">✓</span> : null}
          {o.label}
        </button>
        {allowQty && on && !locked ? (
          <span className="cravr-qty">
            <button
              type="button"
              className="cravr-qty-btn"
              aria-label="Decrease quantity"
              onClick={() => bumpQty(o.value, -1)}
            >
              −
            </button>
            <span className="cravr-qty-val">{qty[o.value] || 1}</span>
            <button
              type="button"
              className="cravr-qty-btn"
              aria-label="Increase quantity"
              onClick={() => bumpQty(o.value, 1)}
            >
              +
            </button>
          </span>
        ) : null}
      </span>
    );
  };

  const groups = groupOptions(options);
  const totalQty = allowQty ? chosen.reduce((sum, v) => sum + (qty[v] || 1), 0) : chosen.length;
  return (
    <div className="cravr-block">
      {groups.map((g, i) => (
        <div className="cravr-group" key={i}>
          {g.name ? (
            <div className="cravr-group-title">
              {g.name}
              {g.required ? <span className="cravr-req">Required</span> : null}
            </div>
          ) : null}
          <div className="cravr-chips">{g.items.map(chip)}</div>
        </div>
      ))}
      {multi && !locked ? (
        <button className="cravr-primary" onClick={submit}>
          {chosen.length
            ? allowQty
              ? `Add ${totalQty} item${totalQty === 1 ? '' : 's'}`
              : `Add ${chosen.length} selected`
            : 'Skip — none of these'}
        </button>
      ) : null}
      {!locked && props.hint ? <div className="cravr-hint">{props.hint}</div> : null}
    </div>
  );
}
