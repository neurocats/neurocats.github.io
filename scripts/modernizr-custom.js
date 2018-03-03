/*! modernizr 3.5.0 (Custom Build) | MIT *
 * https://modernizr.com/download/?-canvas-canvasblending-canvastext-canvaswinding-inlinesvg-smil-svg-svgasimg-svgclippaths-svgfilters-svgforeignobject-todataurljpeg_todataurlpng_todataurlwebp-setclasses !*/
!function(e,t,n){function a(e,t){return typeof e===t}function o(){var e,t,n,o,r,s,i;for(var f in l)if(l.hasOwnProperty(f)){if(e=[],t=l[f],t.name&&(e.push(t.name.toLowerCase()),t.options&&t.options.aliases&&t.options.aliases.length))for(n=0;n<t.options.aliases.length;n++)e.push(t.options.aliases[n].toLowerCase());for(o=a(t.fn,"function")?t.fn():t.fn,r=0;r<e.length;r++)s=e[r],i=s.split("."),1===i.length?Modernizr[i[0]]=o:(!Modernizr[i[0]]||Modernizr[i[0]]instanceof Boolean||(Modernizr[i[0]]=new Boolean(Modernizr[i[0]])),Modernizr[i[0]][i[1]]=o),c.push((o?"":"no-")+i.join("-"))}}function r(e){var t=u.className,n=Modernizr._config.classPrefix||"";if(d&&(t=t.baseVal),Modernizr._config.enableJSClass){var a=new RegExp("(^|\\s)"+n+"no-js(\\s|$)");t=t.replace(a,"$1"+n+"js$2")}Modernizr._config.enableClasses&&(t+=" "+n+e.join(" "+n),d?u.className.baseVal=t:u.className=t)}function s(){return"function"!=typeof t.createElement?t.createElement(arguments[0]):d?t.createElementNS.call(t,"http://www.w3.org/2000/svg",arguments[0]):t.createElement.apply(t,arguments)}function i(e,t){if("object"==typeof e)for(var n in e)v(e,n)&&i(n,e[n]);else{e=e.toLowerCase();var a=e.split("."),o=Modernizr[a[0]];if(2==a.length&&(o=o[a[1]]),"undefined"!=typeof o)return Modernizr;t="function"==typeof t?t():t,1==a.length?Modernizr[a[0]]=t:(!Modernizr[a[0]]||Modernizr[a[0]]instanceof Boolean||(Modernizr[a[0]]=new Boolean(Modernizr[a[0]])),Modernizr[a[0]][a[1]]=t),r([(t&&0!=t?"":"no-")+a.join("-")]),Modernizr._trigger(e,t)}return Modernizr}var c=[],l=[],f={_version:"3.5.0",_config:{classPrefix:"",enableClasses:!0,enableJSClass:!0,usePrefixes:!0},_q:[],on:function(e,t){var n=this;setTimeout(function(){t(n[e])},0)},addTest:function(e,t,n){l.push({name:e,fn:t,options:n})},addAsyncTest:function(e){l.push({name:null,fn:e})}},Modernizr=function(){};Modernizr.prototype=f,Modernizr=new Modernizr,Modernizr.addTest("svg",!!t.createElementNS&&!!t.createElementNS("http://www.w3.org/2000/svg","svg").createSVGRect),Modernizr.addTest("svgfilters",function(){var t=!1;try{t="SVGFEColorMatrixElement"in e&&2==SVGFEColorMatrixElement.SVG_FECOLORMATRIX_TYPE_SATURATE}catch(n){}return t});var u=t.documentElement,d="svg"===u.nodeName.toLowerCase();Modernizr.addTest("canvas",function(){var e=s("canvas");return!(!e.getContext||!e.getContext("2d"))}),Modernizr.addTest("canvastext",function(){return Modernizr.canvas===!1?!1:"function"==typeof s("canvas").getContext("2d").fillText}),Modernizr.addTest("canvasblending",function(){if(Modernizr.canvas===!1)return!1;var e=s("canvas").getContext("2d");try{e.globalCompositeOperation="screen"}catch(t){}return"screen"===e.globalCompositeOperation});var g=s("canvas");Modernizr.addTest("todataurljpeg",function(){return!!Modernizr.canvas&&0===g.toDataURL("image/jpeg").indexOf("data:image/jpeg")}),Modernizr.addTest("todataurlpng",function(){return!!Modernizr.canvas&&0===g.toDataURL("image/png").indexOf("data:image/png")}),Modernizr.addTest("todataurlwebp",function(){var e=!1;try{e=!!Modernizr.canvas&&0===g.toDataURL("image/webp").indexOf("data:image/webp")}catch(t){}return e}),Modernizr.addTest("canvaswinding",function(){if(Modernizr.canvas===!1)return!1;var e=s("canvas").getContext("2d");return e.rect(0,0,10,10),e.rect(2,2,6,6),e.isPointInPath(5,5,"evenodd")===!1}),Modernizr.addTest("inlinesvg",function(){var e=s("div");return e.innerHTML="<svg/>","http://www.w3.org/2000/svg"==("undefined"!=typeof SVGRect&&e.firstChild&&e.firstChild.namespaceURI)});var p={}.toString;Modernizr.addTest("svgclippaths",function(){return!!t.createElementNS&&/SVGClipPath/.test(p.call(t.createElementNS("http://www.w3.org/2000/svg","clipPath")))}),Modernizr.addTest("svgforeignobject",function(){return!!t.createElementNS&&/SVGForeignObject/.test(p.call(t.createElementNS("http://www.w3.org/2000/svg","foreignObject")))}),Modernizr.addTest("smil",function(){return!!t.createElementNS&&/SVGAnimate/.test(p.call(t.createElementNS("http://www.w3.org/2000/svg","animate")))});var v;!function(){var e={}.hasOwnProperty;v=a(e,"undefined")||a(e.call,"undefined")?function(e,t){return t in e&&a(e.constructor.prototype[t],"undefined")}:function(t,n){return e.call(t,n)}}(),f._l={},f.on=function(e,t){this._l[e]||(this._l[e]=[]),this._l[e].push(t),Modernizr.hasOwnProperty(e)&&setTimeout(function(){Modernizr._trigger(e,Modernizr[e])},0)},f._trigger=function(e,t){if(this._l[e]){var n=this._l[e];setTimeout(function(){var e,a;for(e=0;e<n.length;e++)(a=n[e])(t)},0),delete this._l[e]}},Modernizr._q.push(function(){f.addTest=i}),Modernizr.addTest("svgasimg",t.implementation.hasFeature("http://www.w3.org/TR/SVG11/feature#Image","1.1")),o(),r(c),delete f.addTest,delete f.addAsyncTest;for(var m=0;m<Modernizr._q.length;m++)Modernizr._q[m]();e.Modernizr=Modernizr}(window,document);