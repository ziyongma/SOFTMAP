/* ============================================================
   SOFTMAP — Project Page Scripts
   ============================================================ */

document.addEventListener('DOMContentLoaded', function () {

  /* ---- Copy BibTeX button ---- */
  var copyBtn = document.getElementById('copy-bibtex');
  var bibtexCode = document.getElementById('bibtex-code');
  if (copyBtn && bibtexCode) {
    copyBtn.addEventListener('click', function () {
      var text = bibtexCode.textContent;
      navigator.clipboard.writeText(text).then(function () {
        copyBtn.innerHTML = '<span class="icon is-small"><i class="fas fa-check"></i></span>';
        setTimeout(function () {
          copyBtn.innerHTML = '<span class="icon is-small"><i class="fas fa-copy"></i></span>';
        }, 2000);
      }).catch(function () {
        var textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        copyBtn.innerHTML = '<span class="icon is-small"><i class="fas fa-check"></i></span>';
        setTimeout(function () {
          copyBtn.innerHTML = '<span class="icon is-small"><i class="fas fa-copy"></i></span>';
        }, 2000);
      });
    });
  }

});
