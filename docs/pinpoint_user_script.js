// Pinpoint — Vessel Annotation Tool
(function () {
  let pins = [];
  let pinning = false;

  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:99999;pointer-events:none;';
  document.body.appendChild(overlay);

  const tool = document.createElement('div');
  tool.id = 'pinpoint-tool';
  tool.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:100000;background:#1e1e2e;color:#cdd6f4;padding:12px 18px;border-radius:12px;font-family:monospace;font-size:14px;cursor:pointer;border:1px solid #313244;box-shadow:0 8px 24px rgba(0,0,0,0.5);pointer-events:auto;';
  tool.innerHTML = '📍 Pinpoint';
  tool.onclick = togglePinning;
  document.body.appendChild(tool);

  function togglePinning() {
    pinning = !pinning;
    tool.innerHTML = pinning ? '📍 Pinning (click element)' : '📍 Pinpoint';
    tool.style.border = pinning ? '2px solid #89b4fa' : '1px solid #313244';
    document[pinning ? 'addEventListener' : 'removeEventListener']('click', handleClick, true);
  }

  function handleClick(event) {
    if (!pinning || event.target === tool) return;
    event.preventDefault();
    event.stopPropagation();

    const target = event.target;
    const pin = {
      id: 'pin-' + Date.now(),
      x: event.clientX,
      y: event.clientY,
      selector: getSelector(target),
      element_html: target.outerHTML.substring(0, 200),
      comment: prompt('Enter comment for this pin:'),
      timestamp: new Date().toISOString(),
    };

    if (pin.comment !== null) {
      pins.push(pin);
      showPin(pin);
      sendPins();
    }

    pinning = false;
    tool.innerHTML = '📍 Pinpoint';
    tool.style.border = '1px solid #313244';
    document.removeEventListener('click', handleClick, true);
  }

  function getSelector(el) {
    if (el.id) return '#' + el.id;
    if (typeof el.className === 'string') {
      const classes = el.className.split(' ').filter(Boolean).join('.');
      if (classes) return el.tagName.toLowerCase() + '.' + classes;
    }
    const path = [];
    while (el && el.tagName) {
      path.unshift(el.tagName.toLowerCase());
      if (el.id) break;
      el = el.parentElement;
    }
    return path.join(' > ');
  }

  function showPin(pin) {
    const dot = document.createElement('div');
    dot.style.cssText = `position:fixed;top:${pin.y - 8}px;left:${pin.x - 8}px;width:16px;height:16px;background:#89b4fa;border-radius:50%;z-index:100001;pointer-events:none;box-shadow:0 0 12px rgba(137,180,250,0.6);`;
    overlay.appendChild(dot);

    const label = document.createElement('div');
    label.style.cssText = `position:fixed;top:${pin.y + 12}px;left:${pin.x + 12}px;background:#1e1e2e;color:#cdd6f4;padding:6px 12px;border-radius:8px;font-size:12px;font-family:monospace;z-index:100001;pointer-events:none;max-width:300px;border:1px solid #313244;`;
    label.textContent = pin.comment || '📍';
    overlay.appendChild(label);
  }

  function sendPins() {
    fetch('http://localhost:8000/pinpoint/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        page_url: window.location.href,
        page_title: document.title,
        pins,
        user_id: 'vessel-user',
      }),
    }).catch(function () {
      localStorage.setItem('pinpoint_pins', JSON.stringify(pins));
    });
  }

  const saved = localStorage.getItem('pinpoint_pins');
  if (saved) {
    pins = JSON.parse(saved);
    pins.forEach(showPin);
  }

  console.log('📍 Pinpoint loaded. Click the tool to start pinning.');
}());
