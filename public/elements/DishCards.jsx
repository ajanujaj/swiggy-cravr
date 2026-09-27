import { useState } from 'react';

const LOOKS = [
  [/pizza|italian|pasta/i, '🍕', ['#ff7a45', '#e5383b']],
  [/burger|american|fast food/i, '🍔', ['#ffb347', '#ff6a00']],
  [/biryani|hyderabad|mughlai/i, '🍛', ['#f7b733', '#fc4a1a']],
  [/north indian|punjabi|curry/i, '🍛', ['#ff9966', '#ff5e62']],
  [/south indian|dosa|idli/i, '🥞', ['#f6d365', '#fda085']],
  [/chinese|asian|noodle|thai|japanese|sushi|korean/i, '🍜', ['#f857a6', '#ff5858']],
  [/dessert|ice cream|cake|sweet|bakery/i, '🍰', ['#fbc2eb', '#f68084']],
  [/beverage|drink|juice|shake|coffee|tea/i, '🥤', ['#43cea2', '#185a9d']],
  [/salad|healthy|bowl|vegan/i, '🥗', ['#56ab2f', '#a8e063']],
  [/roll|wrap|kebab|shawarma|mediterranean|lebanese/i, '🌯', ['#f2994a', '#eb5757']],
  [/mexican|taco/i, '🌮', ['#f7971e', '#ffd200']],
];

function look(dish) {
  const text = `${dish.cuisine || ''} ${dish.name || ''}`;
  for (const [re, emoji, colors] of LOOKS) if (re.test(text)) return { emoji, colors };
  return { emoji: '🍽️', colors: ['#ff7e45', '#ff5200'] };
}

export default function DishCards() {
  const dishes = props.dishes || [];
  const extras = props.extras || [];
  const [picked, setPicked] = useState(null);
  const locked = !!props.locked || picked !== null;
  const selected = props.selected ?? picked;

  const pick = (o) => {
    if (locked || !o) return;
    setPicked(o.value);
    callAction({ name: 'pick', payload: { value: o.value, label: o.label } });
  };

  return (
    <div className="cravr-block">
      <div className="cravr-rail">
        {dishes.map((d, i) => {
          const { emoji, colors } = look(d);
          const isPicked = d.option && selected === d.option.value;
          const cls = 'cravr-card' + (isPicked ? ' is-picked' : locked && selected != null ? ' is-dim' : '');
          return (
            <div className={cls} key={i}>
              <div className="cravr-card-hero" style={{ background: `linear-gradient(135deg, ${colors[0]}, ${colors[1]})` }}>
                <span className="cravr-card-emoji">{emoji}</span>
                {d.cuisine ? <span className="cravr-card-cuisine">{d.cuisine}</span> : null}
              </div>
              <div className="cravr-card-body">
                <div className="cravr-card-title">
                  {d.veg === true ? <span className="cravr-diet veg" title="Veg" /> : null}
                  {d.veg === false ? <span className="cravr-diet nonveg" title="Non-veg" /> : null}
                  <div className="cravr-card-name">{d.name}</div>
                </div>
                {d.restaurant ? <div className="cravr-card-rest">{d.restaurant}</div> : null}
                <div className="cravr-card-meta">
                  {d.rating != null ? <span className="cravr-rating">★ {d.rating}</span> : null}
                  {d.price != null ? <span className="cravr-price">₹{d.price}</span> : null}
                </div>
                {d.tags && d.tags.length ? (
                  <div className="cravr-tags">
                    {d.tags.slice(0, 3).map((t) => <span className="cravr-tag" key={t}>{t}</span>)}
                  </div>
                ) : null}
                {d.option ? (
                  <button className="cravr-card-cta" disabled={locked} onClick={() => pick(d.option)}>
                    {isPicked ? '✓ Picked' : d.option.cta || d.option.label}
                  </button>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
      {extras.length ? (
        <div className="cravr-chips">
          {extras.map((o) => (
            <button
              key={o.value}
              className={'cravr-chip' + (selected === o.value ? ' is-picked' : '')}
              disabled={locked}
              onClick={() => pick(o)}
            >
              {o.label}
            </button>
          ))}
        </div>
      ) : null}
      {!locked && props.hint ? <div className="cravr-hint">{props.hint}</div> : null}
    </div>
  );
}
