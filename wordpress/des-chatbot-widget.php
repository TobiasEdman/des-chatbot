<?php
/**
 * Plugin Name: DES Chatbot
 * Description: AI-driven chatbot for Digital Earth Sweden
 * Version: 1.0.0
 * Author: Digital Earth Sweden
 * License: GPL-2.0+
 * Text Domain: des-chatbot
 */

if (!defined('ABSPATH')) exit;

// --- Admin Settings ---
add_action('admin_menu', function () {
    add_options_page('DES Chatbot', 'DES Chatbot', 'manage_options', 'des-chatbot', 'des_chatbot_settings_page');
});

add_action('admin_init', function () {
    register_setting('des_chatbot_options', 'des_chatbot_api_url', [
        'type' => 'string',
        'default' => 'https://chat.digitalearth.se/api/chat',
        'sanitize_callback' => 'esc_url_raw',
    ]);
});

function des_chatbot_settings_page() {
    ?>
    <div class="wrap">
        <h1>DES Chatbot Settings</h1>
        <form method="post" action="options.php">
            <?php settings_fields('des_chatbot_options'); ?>
            <table class="form-table">
                <tr>
                    <th scope="row"><label for="des_chatbot_api_url">API URL</label></th>
                    <td>
                        <input type="url" id="des_chatbot_api_url" name="des_chatbot_api_url"
                               value="<?php echo esc_attr(get_option('des_chatbot_api_url', 'https://chat.digitalearth.se/api/chat')); ?>"
                               class="regular-text" />
                        <p class="description">The chat API endpoint URL.</p>
                    </td>
                </tr>
            </table>
            <?php submit_button(); ?>
        </form>
    </div>
    <?php
}

// --- Shortcode ---
add_shortcode('des_chatbot', function () {
    ob_start();
    des_chatbot_render_widget();
    return ob_get_clean();
});

// --- Auto-inject via footer ---
add_action('wp_footer', 'des_chatbot_render_widget');

function des_chatbot_render_widget() {
    static $rendered = false;
    if ($rendered) return;
    $rendered = true;
    $api_url = esc_url(get_option('des_chatbot_api_url', 'https://chat.digitalearth.se/api/chat'));
    ?>
<style>
:root{--des-blue:#1a4a6e;--des-green:#2d8c5a;--des-light:#f0f4f8;--des-radius:12px}
#des-chat-toggle{position:fixed;bottom:24px;right:24px;z-index:99999;width:60px;height:60px;border-radius:50%;background:var(--des-blue);color:#fff;border:none;cursor:pointer;box-shadow:0 4px 16px rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center;transition:transform .2s,background .2s}
#des-chat-toggle:hover{transform:scale(1.08);background:#153d5c}
#des-chat-toggle:focus-visible{outline:3px solid var(--des-green);outline-offset:3px}
#des-chat-toggle svg{width:28px;height:28px;fill:currentColor;transition:transform .2s}
#des-chat-toggle.open svg{transform:rotate(90deg)}
#des-chat-panel{position:fixed;bottom:96px;right:24px;z-index:99999;width:400px;height:500px;max-height:80vh;max-width:calc(100vw - 32px);background:#fff;border-radius:var(--des-radius);box-shadow:0 8px 32px rgba(0,0,0,.18);display:none;flex-direction:column;overflow:hidden;font-family:system-ui,-apple-system,sans-serif}
#des-chat-panel.visible{display:flex}
.des-chat-header{background:var(--des-blue);color:#fff;padding:14px 18px;font-size:15px;font-weight:600;display:flex;align-items:center;gap:10px;flex-shrink:0}
.des-chat-header svg{width:22px;height:22px;fill:currentColor;flex-shrink:0}
.des-chat-messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px;background:var(--des-light)}
.des-msg{max-width:82%;padding:10px 14px;border-radius:var(--des-radius);font-size:14px;line-height:1.55;word-wrap:break-word}
.des-msg a{color:var(--des-blue);text-decoration:underline}
.des-msg code{background:rgba(0,0,0,.07);padding:1px 5px;border-radius:4px;font-size:13px}
.des-msg pre{background:#1e293b;color:#e2e8f0;padding:10px 12px;border-radius:8px;overflow-x:auto;margin:6px 0;font-size:13px}
.des-msg pre code{background:none;padding:0;color:inherit}
.des-msg ul,.des-msg ol{margin:4px 0 4px 18px;padding:0}
.des-msg li{margin:2px 0}
.des-msg-user{background:var(--des-blue);color:#fff;align-self:flex-end;border-bottom-right-radius:4px}
.des-msg-bot{background:#fff;color:#1e293b;align-self:flex-start;border-bottom-left-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.des-typing{align-self:flex-start;background:#fff;padding:10px 18px;border-radius:var(--des-radius);border-bottom-left-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.08);display:none;gap:5px;align-items:center}
.des-typing.visible{display:flex}
.des-typing span{width:7px;height:7px;background:var(--des-blue);border-radius:50%;animation:desBounce 1.2s infinite}
.des-typing span:nth-child(2){animation-delay:.2s}
.des-typing span:nth-child(3){animation-delay:.4s}
@keyframes desBounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-6px)}}
.des-chat-input{display:flex;padding:10px 12px;border-top:1px solid #e2e8f0;background:#fff;gap:8px;flex-shrink:0}
.des-chat-input textarea{flex:1;border:1px solid #cbd5e1;border-radius:8px;padding:9px 12px;font-size:14px;font-family:inherit;resize:none;min-height:20px;max-height:80px;line-height:1.4;outline:none;transition:border-color .15s}
.des-chat-input textarea:focus{border-color:var(--des-blue)}
.des-chat-input button{background:var(--des-blue);color:#fff;border:none;border-radius:8px;width:40px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .15s}
.des-chat-input button:hover{background:#153d5c}
.des-chat-input button:disabled{opacity:.5;cursor:not-allowed}
.des-chat-input button:focus-visible,.des-chat-input textarea:focus-visible{outline:2px solid var(--des-green);outline-offset:1px}
.des-chat-input button svg{width:18px;height:18px;fill:currentColor}
.des-chat-footer{text-align:center;padding:6px 10px;font-size:11px;color:#94a3b8;background:#fff;border-top:1px solid #f1f5f9;flex-shrink:0}
.des-chat-footer a{color:var(--des-green);text-decoration:none}
.des-chat-footer a:hover{text-decoration:underline}
@media(max-width:480px){
  #des-chat-panel{bottom:0;right:0;width:100%;height:100vh;max-height:100vh;max-width:100%;border-radius:0}
  #des-chat-toggle{bottom:16px;right:16px;width:54px;height:54px}
}
</style>

<button id="des-chat-toggle" aria-label="Open chat" aria-expanded="false" title="Chat with us">
  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.2L4 17.2V4h16v12z"/></svg>
</button>

<div id="des-chat-panel" role="dialog" aria-label="DES Chatbot" aria-modal="false">
  <div class="des-chat-header">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>
    DES Chatbot
  </div>
  <div class="des-chat-messages" id="des-chat-messages" role="log" aria-live="polite" aria-label="Chat messages"></div>
  <div class="des-typing" id="des-typing" aria-label="Bot is typing"><span></span><span></span><span></span></div>
  <form class="des-chat-input" id="des-chat-form" aria-label="Send a message">
    <textarea id="des-chat-input" rows="1" placeholder="Type a message..." aria-label="Message input" maxlength="2000"></textarea>
    <button type="submit" aria-label="Send message" id="des-chat-send"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg></button>
  </form>
  <div class="des-chat-footer">Powered by <a href="https://digitalearth.se" target="_blank" rel="noopener">Digital Earth Sweden</a></div>
</div>

<script>
(function(){
  const API_URL = <?php echo wp_json_encode($api_url); ?>;
  const toggle = document.getElementById('des-chat-toggle');
  const panel = document.getElementById('des-chat-panel');
  const msgBox = document.getElementById('des-chat-messages');
  const form = document.getElementById('des-chat-form');
  const input = document.getElementById('des-chat-input');
  const sendBtn = document.getElementById('des-chat-send');
  const typing = document.getElementById('des-typing');
  let busy = false;

  function sessionId(){
    let id = sessionStorage.getItem('des_chat_sid');
    if(!id){id=crypto.randomUUID?crypto.randomUUID():Math.random().toString(36).slice(2)+Date.now().toString(36);sessionStorage.setItem('des_chat_sid',id)}
    return id;
  }

  toggle.addEventListener('click',function(){
    const open = panel.classList.toggle('visible');
    toggle.classList.toggle('open',open);
    toggle.setAttribute('aria-expanded',open);
    toggle.setAttribute('aria-label',open?'Close chat':'Open chat');
    if(open){input.focus();if(!msgBox.children.length)addMsg('bot','Hello! How can I help you today?')}
  });

  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'&&panel.classList.contains('visible')){
      panel.classList.remove('visible');toggle.classList.remove('open');
      toggle.setAttribute('aria-expanded','false');toggle.setAttribute('aria-label','Open chat');
      toggle.focus();
    }
  });

  input.addEventListener('input',function(){this.style.height='auto';this.style.height=Math.min(this.scrollHeight,80)+'px'});

  input.addEventListener('keydown',function(e){
    if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.dispatchEvent(new Event('submit',{cancelable:true}))}
  });

  form.addEventListener('submit',function(e){
    e.preventDefault();
    const text=input.value.trim();
    if(!text||busy)return;
    addMsg('user',text);
    input.value='';input.style.height='auto';
    sendMessage(text);
  });

  function addMsg(role,content){
    const div=document.createElement('div');
    div.className='des-msg des-msg-'+role;
    div.innerHTML=role==='bot'?renderMd(content):escHtml(content);
    msgBox.appendChild(div);
    msgBox.scrollTop=msgBox.scrollHeight;
    return div;
  }

  function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

  function renderMd(s){
    s=escHtml(s);
    // code blocks
    s=s.replace(/```(\w*)\n?([\s\S]*?)```/g,function(_,lang,code){return '<pre><code>'+code.trim()+'</code></pre>'});
    // inline code
    s=s.replace(/`([^`]+)`/g,'<code>$1</code>');
    // bold
    s=s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
    // italic
    s=s.replace(/\*(.+?)\*/g,'<em>$1</em>');
    // links
    s=s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
    // unordered lists
    s=s.replace(/(^|\n)- (.+)/g,function(_,pre,item){return pre+'<li>'+item+'</li>'});
    s=s.replace(/((<li>.*<\/li>\n?)+)/g,'<ul>$1</ul>');
    // ordered lists
    s=s.replace(/(^|\n)\d+\. (.+)/g,function(_,pre,item){return pre+'<li>'+item+'</li>'});
    // line breaks
    s=s.replace(/\n/g,'<br>');
    return s;
  }

  function sendMessage(text){
    busy=true;sendBtn.disabled=true;typing.classList.add('visible');
    msgBox.scrollTop=msgBox.scrollHeight;
    let botDiv=null;let accumulated='';

    const params=new URLSearchParams({message:text,session_id:sessionId()});
    const es=new EventSource(API_URL+'?'+params.toString());

    es.onmessage=function(e){
      if(e.data==='[DONE]'){es.close();finish();return}
      try{
        const d=JSON.parse(e.data);
        const chunk=d.content||d.text||d.delta||d.message||'';
        if(chunk){
          accumulated+=chunk;
          if(!botDiv)botDiv=addMsg('bot','');
          botDiv.innerHTML=renderMd(accumulated);
          msgBox.scrollTop=msgBox.scrollHeight;
        }
      }catch(_){
        accumulated+=e.data;
        if(!botDiv)botDiv=addMsg('bot','');
        botDiv.innerHTML=renderMd(accumulated);
        msgBox.scrollTop=msgBox.scrollHeight;
      }
    };

    es.onerror=function(){
      es.close();
      if(!accumulated){addMsg('bot','Sorry, something went wrong. Please try again.')}
      finish();
    };

    function finish(){busy=false;sendBtn.disabled=false;typing.classList.remove('visible');input.focus()}
  }
})();
</script>
    <?php
}
