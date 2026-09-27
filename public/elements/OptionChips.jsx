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
  const [picked, setPicked] = useState(null);
  const [chosen, setChosen] = useState([]);
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
    setChosen((c) => (c.includes(o.value) ? c.filter((v) => v !== o.value) : [...c, o.value]));
  };
  const submit = () => {
    if (locked) return;
    setPicked(chosen);
    callAction({ name: 'multiselect_submit', payload: { values: chosen } });
  };

  const chip = (o) => {
    const on = multi ? selectedList.includes(o.value) : selected === o.value;
    const cls = 'cravr-chip' + (on ? (multi ? ' is-on' : ' is-picked') : '');
    return (
      <button key={o.value} className={cls} disabled={locked} onClick={() => (multi ? toggle(o) : pick(o))}>
        {multi && on ? <span className="cravr-tick">✓</span> : null}
        {o.label}
      </button>
    );
  };

  const groups = groupOptions(options);
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
          {chosen.length ? `Add ${chosen.length} selected` : 'Skip — none of these'}
        </button>
      ) : null}
      {!locked && props.hint ? <div className="cravr-hint">{props.hint}</div> : null}
    </div>
  );
}
