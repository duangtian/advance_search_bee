# Simple HTTP Server for TSP Web UI
$port = 8080
$path = "c:\Users\pepsi\Documents\WorkSpace\bee\advance_search_bee\web_ui"

Write-Host "🌐 Starting TSP Solver Web Server..." -ForegroundColor Green
Write-Host "📁 Serving files from: $path" -ForegroundColor Cyan
Write-Host "🔗 Open your browser to: http://localhost:$port" -ForegroundColor Yellow
Write-Host "⚠️  Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Start the built-in PowerShell web server
try {
    Add-Type -AssemblyName System.Net.Http
    
    $listener = New-Object System.Net.HttpListener
    $listener.Prefixes.Add("http://localhost:$port/")
    $listener.Start()
    
    Write-Host "✅ Server started successfully on port $port" -ForegroundColor Green
    Write-Host ""
    
    # Open browser automatically
    Start-Process "http://localhost:$port"
    
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response
        
        $requestedFile = $request.Url.LocalPath.TrimStart('/')
        if ($requestedFile -eq "" -or $requestedFile -eq "/") {
            $requestedFile = "index.html"
        }
        
        $filePath = Join-Path $path $requestedFile
        
        Write-Host "📄 Request: $($request.HttpMethod) $($request.Url.LocalPath)" -ForegroundColor Gray
        
        if (Test-Path $filePath) {
            $content = Get-Content $filePath -Raw -Encoding UTF8
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($content)
            
            # Set content type based on file extension
            $extension = [System.IO.Path]::GetExtension($filePath).ToLower()
            switch ($extension) {
                ".html" { $response.ContentType = "text/html; charset=utf-8" }
                ".css" { $response.ContentType = "text/css; charset=utf-8" }
                ".js" { $response.ContentType = "application/javascript; charset=utf-8" }
                ".json" { $response.ContentType = "application/json; charset=utf-8" }
                default { $response.ContentType = "text/plain; charset=utf-8" }
            }
            
            $response.ContentLength64 = $bytes.Length
            $response.StatusCode = 200
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        } else {
            $response.StatusCode = 404
            $errorBytes = [System.Text.Encoding]::UTF8.GetBytes("404 - File not found: $requestedFile")
            $response.ContentLength64 = $errorBytes.Length
            $response.OutputStream.Write($errorBytes, 0, $errorBytes.Length)
            Write-Host "❌ File not found: $filePath" -ForegroundColor Red
        }
        
        $response.Close()
    }
} catch {
    Write-Host "❌ Error starting server: $($_.Exception.Message)" -ForegroundColor Red
    if ($listener -and $listener.IsListening) {
        $listener.Stop()
    }
} finally {
    if ($listener) {
        $listener.Close()
    }
    Write-Host "🛑 Server stopped" -ForegroundColor Yellow
}