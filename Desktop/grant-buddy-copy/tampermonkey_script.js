// ==UserScript==
// @name         GrantBuddy Helper – SAM.gov + SBIR.gov (Fixed)
// @namespace    http://tampermonkey.net/
// @version      2.3
// @description  Auto extraction for SAM.gov, Highlight-to-scrape for SBIR.gov
// @match        https://sam.gov/opp/*
// @match        https://sam.gov/workspace/contract/opp/*
// @match        https://www.sam.gov/opp/*
// @match        https://www.sam.gov/workspace/contract/opp/*
// @match        https://www.dodsbirsttr.mil/topics-app/*
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    const isSAM = window.location.hostname.includes('sam.gov') && 
                  (window.location.pathname.includes('/opp/') || 
                   window.location.pathname.includes('/workspace/contract/opp/'));
    const isSBIR = window.location.href.includes('dodsbirsttr.mil/topics-app');

    console.log('GrantBuddy: Script loaded', { isSAM, isSBIR, url: window.location.href });

    // ──────────────────────────────
    // 🟦 SAM.gov Wing
    // ──────────────────────────────
    if (isSAM) {
        // Wait longer for Angular app to load
        setTimeout(() => {
            console.log('GrantBuddy: First injection attempt');
            if (!document.getElementById('grantbuddy-btn-container')) {
                injectSAM();
            }
        }, 5000);

        // Keep trying every few seconds
        const retryInterval = setInterval(() => {
            if (!document.getElementById('grantbuddy-btn-container')) {
                console.log('GrantBuddy: Retry injection attempt');
                injectSAM();
            } else {
                clearInterval(retryInterval);
            }
        }, 3000);

        function injectSAM() {
            console.log('GrantBuddy: Attempting to extract SAM.gov data...');

            // Check if Angular content is loaded
            const appRoot = document.querySelector('app-opps-display');
            if (!appRoot) {
                console.log('GrantBuddy: Angular app not ready yet');
                return;
            }

            let title = '';
            let noticeId = '';
            let agency = '';
            let responseDate = '';

            // Extract title - look for the specific h1 with card-title class
            const titleElement = document.querySelector('h1.card-title');
            if (titleElement) {
                title = titleElement.textContent.trim();
            }

            // Extract Notice ID - look for h5 with aria-describedby="notice-id"
            const noticeElement = document.querySelector('h5[aria-describedby="notice-id"]');
            if (noticeElement) {
                noticeId = noticeElement.textContent.trim();
            }

            // Extract Response Date - look for h5 with aria-describedby="date-offers-date"
            const responseElement = document.querySelector('h5[aria-describedby="date-offers-date"]');
            if (responseElement) {
                responseDate = responseElement.textContent.trim();
            }

            // Extract Department/Agency - look for h5 with aria-describedby="dept-agency"
            const deptElement = document.querySelector('h5[aria-describedby="dept-agency"]');
            if (deptElement) {
                agency = deptElement.textContent.trim();
            }

            // Extract description - look for span with aria-describedby="desc"
            let description = '';
            const descElement = document.querySelector('span[aria-describedby="desc"]');
            if (descElement) {
                description = descElement.textContent.trim().substring(0, 3000);
            }

            const extractedData = [
                `Title: ${title || '(No title found)'}`,
                `Notice ID: ${noticeId || '(Not found)'}`,
                `Agency: ${agency || '(Not found)'}`,
                `Response Date: ${responseDate || '(Not found)'}`,
                '',
                'Description:',
                description || '(No description found)',
                '',
                `Source URL: ${window.location.href}`,
                `Extracted at: ${new Date().toISOString()}`
            ].join('\n');

            createFloatingButton(extractedData);
            console.log('GrantBuddy: Data extracted successfully', {
                title: title ? 'Found' : 'Not found',
                noticeId: noticeId ? 'Found' : 'Not found',
                agency: agency ? 'Found' : 'Not found',
                responseDate: responseDate ? 'Found' : 'Not found'
            });
        }

        function createFloatingButton(extractedData) {
            const existing = document.getElementById('grantbuddy-btn-container');
            if (existing) existing.remove();

            const container = document.createElement('div');
            container.id = 'grantbuddy-btn-container';
            container.style.cssText = `
                position: fixed !important;
                bottom: 20px !important;
                right: 20px !important;
                z-index: 99999 !important;
                background: #fff !important;
                border: 2px solid #005bbb !important;
                padding: 12px !important;
                border-radius: 10px !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
                font-family: 'Be Vietnam Pro', Arial, sans-serif !important;
                max-width: 320px !important;
            `;

            const button = document.createElement('button');
            button.textContent = '📤 Send to GrantBuddy';
            button.style.cssText = `
                background: #005bbb !important;
                color: white !important;
                border: none !important;
                border-radius: 6px !important;
                padding: 10px 16px !important;
                cursor: pointer !important;
                width: 100% !important;
                font-family: 'Be Vietnam Pro', Arial, sans-serif !important;
                font-size: 14px !important;
                font-weight: 500 !important;
                margin-bottom: 8px !important;
            `;

            button.onclick = () => {
                const url = 'http://localhost:8501/?tab=samgov&source=samgov&text=' +
                    encodeURIComponent(extractedData);
                window.open(url, '_blank', 'noopener,noreferrer');
            };

            const info = document.createElement('div');
            info.style.cssText = 'font-size: 12px; color: #666; margin-top: 4px;';
            info.textContent = `✅ Data extracted from SAM.gov`;

            container.appendChild(button);
            container.appendChild(info);
            document.body.appendChild(container);
            console.log('GrantBuddy: Button injected successfully');
        }
    }

    // ──────────────────────────────
    // 🟧 SBIR.gov Wing
    // ──────────────────────────────
    if (isSBIR) {
        function injectUI_SBIR() {
            if (document.getElementById('grantbuddy-sbir-box')) return true;

            const anchor = document.querySelector('app-root');
            if (!anchor) return false;

            const container = document.createElement('div');
            container.id = 'grantbuddy-sbir-box';
            container.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 9999;
                background: #fff;
                border: 2px solid #005bbb;
                padding: 12px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                font-family: 'Be Vietnam Pro', Arial, sans-serif;
                max-width: 300px;
            `;

            const button = document.createElement('button');
            button.textContent = '📤 Analyze Highlighted Text';
            button.style.cssText = `
                background: #005bbb;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                cursor: pointer;
                margin-bottom: 10px;
                width: 100%;
                font-family: 'Be Vietnam Pro', Arial, sans-serif;
                font-size: 14px;
                font-weight: 500;
            `;

            const note = document.createElement('p');
            note.innerHTML = "<b>How to use:</b><br>1. Highlight text on the page.<br>2. Click this button.";
            note.style.cssText = 'font-size: 12px; margin: 0; line-height: 1.4; color: #666;';

            button.onclick = () => {
                const selectedText = window.getSelection().toString().trim();
                if (!selectedText) {
                    alert("⚠️ No text selected.\n\nPlease highlight the topic text you want to analyze before clicking the button.");
                    return;
                }
                const encoded = encodeURIComponent(selectedText.slice(0, 5000));
                const targetURL = `http://localhost:8501/?tab=sbirgov&source=sbirgov&text=${encoded}`;
                window.open(targetURL, '_blank', 'noopener,noreferrer');
            };

            container.appendChild(button);
            container.appendChild(note);
            document.body.appendChild(container);
            return true;
        }

        let attempts = 0;
        const maxAttempts = 30;
        const injectionInterval = setInterval(() => {
            if (injectUI_SBIR() || attempts++ >= maxAttempts) {
                clearInterval(injectionInterval);
            }
        }, 500);
    }
})(); 