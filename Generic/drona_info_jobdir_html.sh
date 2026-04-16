#!/bin/bash

cat <<HTML
<html><head>
<style>
  .dir-box {
    display: inline-block;
    background-color: #f5f5f5;
    border-radius: 4px;
    padding: 6px 10px;
    font-family: monospace;
  }
  a { color: #003c71  ; text-decoration: none;font-weight: bold; }
  a:hover { text-decoration: underline; }
</style>
</head><body>
      <span class="dir-box">
        <a href="/pun/sys/dashboard/files/fs$JOB_DIR" target="_blank">$JOB_DIR</a>
      </span>
</body></html>
HTML
