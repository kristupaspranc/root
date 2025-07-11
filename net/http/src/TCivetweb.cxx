// Author: Sergey Linev   21/12/2013

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TCivetweb.h"

#include <cstdlib>
#include <cstring>

#ifdef _MSC_VER
#include <windows.h>
#include <tchar.h>
#endif

#include "THttpServer.h"
#include "THttpWSEngine.h"
#include "TUrl.h"
#include "TSystem.h"
#include "TError.h"

//////////////////////////////////////////////////////////////////////////
/// TCivetwebWSEngine
///
/// Implementation of THttpWSEngine for Civetweb

class TCivetwebWSEngine : public THttpWSEngine {
protected:
   struct mg_connection *fWSconn;

   /// True websocket requires extra thread to parallelize sending
   Bool_t SupportSendThrd() const override { return kTRUE; }

public:
   TCivetwebWSEngine(struct mg_connection *conn) : THttpWSEngine(), fWSconn(conn) {}

   ~TCivetwebWSEngine() override = default;

   UInt_t GetId() const override { return TString::Hash((void *)&fWSconn, sizeof(void *)); }

   void ClearHandle(Bool_t terminate) override
   {
      if (fWSconn && terminate)
         mg_websocket_write(fWSconn, MG_WEBSOCKET_OPCODE_CONNECTION_CLOSE, nullptr, 0);
      fWSconn = nullptr;
   }

   void Send(const void *buf, int len) override
   {
      if (fWSconn)
         mg_websocket_write(fWSconn, MG_WEBSOCKET_OPCODE_BINARY, (const char *)buf, len);
   }

   /////////////////////////////////////////////////////////
   /// Special method to send binary data with text header
   /// For normal websocket it is two separated operation, for other engines could be combined together,
   /// but emulates as two messages on client side
   void SendHeader(const char *hdr, const void *buf, int len) override
   {
      if (fWSconn) {
         mg_websocket_write(fWSconn, MG_WEBSOCKET_OPCODE_TEXT, hdr, strlen(hdr));
         mg_websocket_write(fWSconn, MG_WEBSOCKET_OPCODE_BINARY, (const char *)buf, len);
      }
   }

   void SendCharStar(const char *str) override
   {
      if (fWSconn)
         mg_websocket_write(fWSconn, MG_WEBSOCKET_OPCODE_TEXT, str, strlen(str));
   }
};

//////////////////////////////////////////////////////////////////////////
/// Check if engine has enough threads to process connect to new websocket handle

Bool_t CheckEngineThreads(TCivetweb *engine, const char *uri, Bool_t longpoll)
{
   Int_t num_avail = engine->GetNumAvailableThreads();
   if (longpoll) num_avail++;

   if ((num_avail <= 0.1 * engine->GetNumThreads()) || (num_avail <= 2)) {
      const char *cfg = engine->IsWebGui() ? "WebGui.HttpThreads parameter in rootrc" : "thrds=N parameter in config URL";
      const char *place = longpoll ? "TCivetweb::LongpollHandler" : "TCivetweb::WebSocketHandler";
      ::Error(place, "Only %d threads are available, reject connection request for %s. Increase %s, now it is %d", num_avail, uri, cfg, engine->GetNumThreads());
      return kFALSE;
   }

   return kTRUE;
}

//////////////////////////////////////////////////////////////////////////

int websocket_connect_handler(const struct mg_connection *conn, void *)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);
   if (!request_info)
      return 1;

   TCivetweb *engine = (TCivetweb *)request_info->user_data;
   if (!engine || engine->IsTerminating())
      return 1;
   THttpServer *serv = engine->GetServer();
   if (!serv)
      return 1;

   auto arg = std::make_shared<THttpCallArg>();
   arg->SetPathAndFileName(request_info->local_uri); // path and file name
   arg->SetQuery(request_info->query_string);        // query arguments
   arg->SetTopName(engine->GetTopName());
   arg->SetWSId(TString::Hash((void *)&conn, sizeof(void *)));
   arg->SetMethod("WS_CONNECT");

   if (!CheckEngineThreads(engine, arg->GetPathName(), kFALSE))
      return 1;

   Bool_t execres = serv->ExecuteWS(arg, kTRUE, kTRUE);

   return execres && !arg->Is404() ? 0 : 1;
}

//////////////////////////////////////////////////////////////////////////

void websocket_ready_handler(struct mg_connection *conn, void *)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);

   TCivetweb *engine = (TCivetweb *)request_info->user_data;
   if (!engine || engine->IsTerminating())
      return;
   THttpServer *serv = engine->GetServer();
   if (!serv)
      return;

   engine->ChangeNumActiveThrerads(1);

   auto arg = std::make_shared<THttpCallArg>();
   arg->SetPathAndFileName(request_info->local_uri); // path and file name
   arg->SetQuery(request_info->query_string);        // query arguments
   arg->SetTopName(engine->GetTopName());
   arg->SetMethod("WS_READY");

   // delegate ownership to the arg, id will be automatically set
   arg->CreateWSEngine<TCivetwebWSEngine>(conn);

   serv->ExecuteWS(arg, kTRUE, kTRUE);
}


//////////////////////////////////////////////////////////////////////////

void websocket_close_handler(const struct mg_connection *conn, void *)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);

   // check if connection was already closed
   if (mg_get_user_connection_data(conn) == (void *) conn)
      return;

   TCivetweb *engine = (TCivetweb *)request_info->user_data;
   if (!engine || engine->IsTerminating())
      return;
   THttpServer *serv = engine->GetServer();
   if (!serv)
      return;

   auto arg = std::make_shared<THttpCallArg>();
   arg->SetPathAndFileName(request_info->local_uri); // path and file name
   arg->SetQuery(request_info->query_string);        // query arguments
   arg->SetTopName(engine->GetTopName());
   arg->SetWSId(TString::Hash((void *)&conn, sizeof(void *)));
   arg->SetMethod("WS_CLOSE");

   serv->ExecuteWS(arg, kTRUE, kFALSE); // do not wait for result of execution

   engine->ChangeNumActiveThrerads(-1);
}

//////////////////////////////////////////////////////////////////////////

int websocket_data_handler(struct mg_connection *conn, int code, char *data, size_t len, void *)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);

   // check if connection data set to connection itself - means connection was closed already
   std::string *conn_data = (std::string *) mg_get_user_connection_data(conn);
   if ((void *) conn_data == (void *) conn)
      return 1;

   // see https://datatracker.ietf.org/doc/html/rfc6455#section-5.2
   int fin = code & 0x80, opcode = code & 0x0F;

   // recognized operation codes, all other should fails
   enum { OP_CONTINUE = 0, OP_TEXT = 1, OP_BINARY = 2, OP_CLOSE = 8 };

   // close when normal close is detected
   if (fin && (opcode == OP_CLOSE)) {
      if (conn_data) delete conn_data;
      websocket_close_handler(conn, nullptr);
      mg_set_user_connection_data(conn, conn); // mark connection as closed
      return 1;
   }

   // ignore empty data
   if (len == 0)
      return 1;

   // close connection when unrecognized opcode is detected
   if ((opcode != OP_CONTINUE) && (opcode != OP_TEXT) && (opcode != OP_BINARY)) {
      if (conn_data) delete conn_data;
      websocket_close_handler(conn, nullptr);
      mg_set_user_connection_data(conn, conn); // mark connection as closed
      return 1;
   }

   TCivetweb *engine = (TCivetweb *)request_info->user_data;
   if (!engine || engine->IsTerminating())
      return 1;
   THttpServer *serv = engine->GetServer();
   if (!serv)
      return 1;

   // this is continuation of the request
   if (!fin) {
      if (!conn_data) {
         conn_data = new std::string(data,len);
         mg_set_user_connection_data(conn, conn_data);
      } else {
         conn_data->append(data,len);
      }
      return 1;
   }

   auto arg = std::make_shared<THttpCallArg>();
   arg->SetPathAndFileName(request_info->local_uri); // path and file name
   arg->SetQuery(request_info->query_string);        // query arguments
   arg->SetTopName(engine->GetTopName());
   arg->SetWSId(TString::Hash((void *)&conn, sizeof(void *)));
   arg->SetMethod("WS_DATA");

   if (conn_data) {
      mg_set_user_connection_data(conn, nullptr);
      conn_data->append(data,len);
      arg->SetPostData(std::move(*conn_data));
      delete conn_data;
   } else {
      arg->SetPostData(std::string(data,len));
   }

   serv->ExecuteWS(arg, kTRUE, kTRUE);

   return 1;
}


//////////////////////////////////////////////////////////////////////////

static int log_message_handler(const struct mg_connection *conn, const char *message)
{
   const struct mg_context *ctx = mg_get_context(conn);

   TCivetweb *engine = (TCivetweb *)mg_get_user_data(ctx);

   if (engine)
      return engine->ProcessLog(message);

   // provide debug output
   if ((gDebug > 0) || strstr(message, "cannot bind to"))
      fprintf(stderr, "Error in <TCivetweb::Log> %s\n", message);

   return 0;
}

struct TEngineHolder {
   TCivetweb *fEngine{nullptr};
   TEngineHolder(TCivetweb *engine)
   {
      fEngine = engine;
      fEngine->ChangeNumActiveThrerads(1);
   }
   ~TEngineHolder()
   {
      fEngine->ChangeNumActiveThrerads(-1);
   }
};

//////////////////////////////////////////////////////////////////////////
/// Returns kTRUE in case of longpoll connection request - or at least looks like that

Bool_t IsBadLongPollConnect(TCivetweb *engine, const std::shared_ptr<THttpCallArg> &arg)
{
   if (strcmp(arg->GetFileName(), "root.longpoll") != 0)
      return kFALSE;

   const char *q = arg->GetQuery();
   if (!q || !*q)
      return kFALSE;

   if ((strstr(q, "raw_connect") != q) && (strstr(q, "txt_connect") != q))
      return kFALSE;

   return !CheckEngineThreads(engine, arg->GetPathName(), kTRUE);
}

//////////////////////////////////////////////////////////////////////////

static int begin_request_handler(struct mg_connection *conn, void *)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);

   TCivetweb *engine = (TCivetweb *)request_info->user_data;
   if (!engine || engine->IsTerminating())
      return 0;
   THttpServer *serv = engine->GetServer();
   if (!serv)
      return 0;

   TEngineHolder thrd_cnt_holder(engine);

   auto arg = std::make_shared<THttpCallArg>();

   TString filename;

   Bool_t execres = kTRUE, debug = engine->IsDebugMode();

   if (!debug && serv->IsFileRequested(request_info->local_uri, filename)) {
      if ((filename.Length() > 3) && ((filename.Index(".js") != kNPOS) || (filename.Index(".css") != kNPOS))) {
         std::string buf = THttpServer::ReadFileContent(filename.Data());
         if (buf.empty()) {
            arg->Set404();
         } else {
            arg->SetContentType(THttpServer::GetMimeType(filename.Data()));
            arg->SetContent(std::move(buf));
            if (engine->GetMaxAge() > 0)
               arg->AddHeader("Cache-Control", TString::Format("max-age=%d", engine->GetMaxAge()));
            else
               arg->AddNoCacheHeader();
            arg->SetZipping();
         }
      } else {
         arg->SetFile(filename.Data());
      }
   } else {
      arg->SetPathAndFileName(request_info->local_uri); // path and file name
      arg->SetQuery(request_info->query_string);        // query arguments
      arg->SetTopName(engine->GetTopName());
      arg->SetMethod(request_info->request_method); // method like GET or POST
      if (request_info->remote_user)
         arg->SetUserName(request_info->remote_user);

      TString header;
      for (int n = 0; n < request_info->num_headers; n++)
         header.Append(
            TString::Format("%s: %s\r\n", request_info->http_headers[n].name, request_info->http_headers[n].value));
      arg->SetRequestHeader(header);

      const char *len = mg_get_header(conn, "Content-Length");
      Int_t ilen = len ? TString(len).Atoi() : 0;

      if (ilen > 0) {
         std::string buf;
         buf.resize(ilen);
         Int_t iread = mg_read(conn, (void *) buf.data(), ilen);
         if (iread == ilen)
            arg->SetPostData(std::move(buf));
      }

      if (debug) {
         TString cont;
         cont.Append("<title>Civetweb echo</title>");
         cont.Append("<h1>Civetweb echo</h1>\n");

         static int count = 0;

         cont.Append(TString::Format("Request %d:<br/>\n<pre>\n", ++count));
         cont.Append(TString::Format("  Method   : %s\n", arg->GetMethod()));
         cont.Append(TString::Format("  PathName : %s\n", arg->GetPathName()));
         cont.Append(TString::Format("  FileName : %s\n", arg->GetFileName()));
         cont.Append(TString::Format("  Query    : %s\n", arg->GetQuery()));
         cont.Append(TString::Format("  PostData : %ld\n", arg->GetPostDataLength()));
         if (arg->GetUserName())
            cont.Append(TString::Format("  User     : %s\n", arg->GetUserName()));

         cont.Append("</pre><p>\n");

         cont.Append("Environment:<br/>\n<pre>\n");
         for (int n = 0; n < request_info->num_headers; n++)
            cont.Append(
               TString::Format("  %s = %s\n", request_info->http_headers[n].name, request_info->http_headers[n].value));
         cont.Append("</pre><p>\n");

         arg->SetContentType("text/html");

         arg->SetContent(cont);

      } else if (IsBadLongPollConnect(engine, arg)) {
         execres = kFALSE;
         arg->Set404();
      } else {
         execres = serv->ExecuteHttp(arg);
      }
   }

   if (!execres || arg->Is404()) {
      std::string hdr = arg->FillHttpHeader("HTTP/1.1");
      mg_printf(conn, "%s", hdr.c_str());
   } else if (arg->IsFile()) {
      filename = (const char *)arg->GetContent();
#ifdef _MSC_VER
      if (engine->IsWinSymLinks()) {
         // resolve Windows links which are not supported by civetweb
         auto hFile = CreateFile(filename.Data(),       // file to open
                                 GENERIC_READ,          // open for reading
                                 FILE_SHARE_READ,       // share for reading
                                 NULL,                  // default security
                                 OPEN_EXISTING,         // existing file only
                                 FILE_ATTRIBUTE_NORMAL, // normal file
                                 NULL);                 // no attr. template

         if( hFile != INVALID_HANDLE_VALUE) {
            const int BUFSIZE = 2048;
            TCHAR Path[BUFSIZE];
            auto dwRet = GetFinalPathNameByHandle( hFile, Path, BUFSIZE, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS );
            // produced file name may include \\?\ symbols, which are indicating long file name
            if(dwRet < BUFSIZE) {
               if (dwRet > 4 && Path[0] == '\\' && Path[1] == '\\' && Path[2] == '?' && Path[3] == '\\')
                  filename = Path + 4;
               else
                  filename = Path;
            }
            CloseHandle(hFile);
         }
      }
#endif
      const char *mime_type = THttpServer::GetMimeType(filename.Data());
      if (mime_type)
         mg_send_mime_file(conn, filename.Data(), mime_type);
      else
         mg_send_file(conn, filename.Data());
   } else {

      Bool_t dozip = kFALSE;
      switch (arg->GetZipping()) {
      case THttpCallArg::kNoZip: dozip = kFALSE; break;
      case THttpCallArg::kZipLarge:
         if (arg->GetContentLength() < 10000) break;
      case THttpCallArg::kZip:
         // check if request header has Accept-Encoding
         for (int n = 0; n < request_info->num_headers; n++) {
            TString name = request_info->http_headers[n].name;
            if (name.Index("Accept-Encoding", 0, TString::kIgnoreCase) != 0)
               continue;
            TString value = request_info->http_headers[n].value;
            dozip = (value.Index("gzip", 0, TString::kIgnoreCase) != kNPOS);
            break;
         }

         break;
      case THttpCallArg::kZipAlways: dozip = kTRUE; break;
      }

      if (dozip)
         arg->CompressWithGzip();

      std::string hdr = arg->FillHttpHeader("HTTP/1.1");
      mg_printf(conn, "%s", hdr.c_str());

      if (arg->GetContentLength() > 0)
         mg_write(conn, arg->GetContent(), (size_t)arg->GetContentLength());
   }

   // Returning non-zero tells civetweb that our function has replied to
   // the client, and civetweb should not send client any more data.
   return 1;
}

/** \class TCivetweb
\ingroup http

THttpEngine implementation, based on civetweb embedded server

It is default kind of engine, created for THttpServer
Currently v1.15 from https://github.com/civetweb/civetweb is used

Additional options can be specified:

    top=foldername - name of top folder, seen in the browser
    thrds=N        - use N threads to run civetweb server (default 5)
    auth_file      - global authentication file
    auth_domain    - domain name, used for authentication

Example:

    new THttpServer("http:8080?top=MyApp&thrds=3");

For the full list of supported options see TCivetweb::Create() documentation

When `auth_file` and `auth_domain` parameters are specified, access
to running http server will be possible only after user
authentication, using so-call digest method. To generate
authentication file, htdigest routine should be used:

    [shell] htdigest -c .htdigest domain_name user

When creating server, parameters should be:

   auto serv = new THttpServer("http:8080?auth_file=.htdigets&auth_domain=domain_name");

*/

////////////////////////////////////////////////////////////////////////////////
/// constructor

TCivetweb::TCivetweb(Bool_t only_secured)
   : THttpEngine("civetweb", "compact embedded http server"),
     fOnlySecured(only_secured)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TCivetweb::~TCivetweb()
{
   if (fCtx && !fTerminating)
      mg_stop(fCtx);
}

////////////////////////////////////////////////////////////////////////////////
/// process civetweb log message, can be used to detect critical errors

Int_t TCivetweb::ProcessLog(const char *message)
{
   if ((gDebug > 0) || (strstr(message, "cannot bind to") != 0))
      Error("Log", "%s", message);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns number of actively used threads

Int_t TCivetweb::GetNumAvailableThreads()
{
   std::lock_guard<std::mutex> guard(fMutex);
   return fNumThreads - fNumActiveThreads;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns number of actively used threads

Int_t TCivetweb::ChangeNumActiveThrerads(int cnt)
{
   std::lock_guard<std::mutex> guard(fMutex);
   fNumActiveThreads += cnt;
   return fNumActiveThreads;
}


////////////////////////////////////////////////////////////////////////////////
/// Creates embedded civetweb server
///
/// @param args string with civetweb server configuration
///
/// As main argument, http port should be specified like "8090".
/// Or one can provide combination of ipaddress and portnumber like "127.0.0.1:8090"
/// Or one can specify unix socket name like "x/tmp/root.socket"
/// Extra parameters like in URL string could be specified after '?' mark:
///
///     thrds=N               - there N is number of threads used by the civetweb (default is 10)
///     top=name              - configure top name, visible in the web browser
///     ssl_certificate=filename - SSL certificate, see docs/OpenSSL.md from civetweb
///     auth_file=filename    - authentication file name, created with htdigets utility
///     auth_domain=domain    - authentication domain
///     websocket_timeout=tm  - set web sockets timeout in seconds (default 300)
///     websocket_disable     - disable web sockets handling (default enabled)
///     bind                  - ip address to bind server socket
///     loopback              - bind specified port to loopback 127.0.0.1 address
///     debug                 - enable debug mode, server always returns html page with request info
///     log=filename          - configure civetweb log file
///     max_age=value         - configures "Cache-Control: max_age=value" http header for all file-related requests, default 3600
///     socket_mode=value     - configures unix socket mode, default is 0700
///     nocache               - try to fully disable cache control for file requests
///     winsymlinks=no        - do not resolve symbolic links on file system (Windows only), default true
///     dirlisting=no         - enable/disable directory listing for browsing filesystem (default no)
///
/// Examples of valid args values:
///
///     serv->CreateEngine("http:8080?websocket_disable");
///     serv->CreateEngine("http:7546?thrds=30&websocket_timeout=20");

Bool_t TCivetweb::Create(const char *args)
{
   memset(&fCallbacks, 0, sizeof(struct mg_callbacks));
   // fCallbacks.begin_request = begin_request_handler;
   fCallbacks.log_message = log_message_handler;
   TString sport = IsSecured() ? "8480s" : "8080",
           num_threads,
           websocket_timeout = "300000",
           dir_listening = "no",
           auth_file,
           auth_domain,
           log_file,
           ssl_cert,
           max_age;
   Int_t socket_mode = 0700;
   bool use_ws = kTRUE, is_socket = false;

   // extract arguments
   if (args && *args) {

      // first extract port number
      sport = "";

      is_socket = *args == 'x';

      while ((*args != 0) && (*args != '?') && (is_socket || (*args != '/')))
         sport.Append(*args++);
      if (IsSecured() && (sport.Index("s") == kNPOS) && !is_socket)
         sport.Append("s");

      // than search for extra parameters
      while ((*args != 0) && (*args != '?'))
         args++;

      if (*args == '?') {
         TUrl url(TString::Format("http://localhost/folder%s", args));

         if (url.IsValid()) {
            url.ParseOptions();

            fWebGui = url.HasOption("webgui");

            const char *top = url.GetValueFromOptions("top");
            if (top)
               fTopName = top;

            const char *log = url.GetValueFromOptions("log");
            if (log)
               log_file = log;

            Int_t thrds = url.GetIntValueFromOptions("thrds");
            if (thrds > 0)
               fNumThreads = thrds;

            const char *afile = url.GetValueFromOptions("auth_file");
            if (afile)
               auth_file = afile;

            const char *adomain = url.GetValueFromOptions("auth_domain");
            if (adomain)
               auth_domain = adomain;

            const char *sslc = url.GetValueFromOptions("ssl_cert");
            if (sslc)
               ssl_cert = sslc;

            Int_t wtmout = url.GetIntValueFromOptions("websocket_timeout");
            if (wtmout > 0) {
               websocket_timeout.Format("%d", wtmout * 1000);
               use_ws = kTRUE;
            }

            if (url.HasOption("websocket_disable"))
               use_ws = kFALSE;

            if (url.HasOption("debug"))
               fDebug = kTRUE;

            const char *winsymlinks = url.GetValueFromOptions("winsymlinks");
            if (winsymlinks)
               fWinSymLinks = strcmp(winsymlinks,"no") != 0;

            const char *dls = url.GetValueFromOptions("dirlisting");
            if (dls && (!strcmp(dls,"no") || !strcmp(dls,"yes")))
               dir_listening = dls;

            const char *smode = url.GetValueFromOptions("socket_mode");
            if (smode)
               socket_mode = std::stoi(smode, nullptr, *smode=='0' ? 8 : 10);

            if (url.HasOption("loopback") && (sport.Index(":") == kNPOS))
               sport = TString("127.0.0.1:") + sport;

            if (url.HasOption("bind") && (sport.Index(":") == kNPOS)) {
               const char *addr = url.GetValueFromOptions("bind");
               if (addr && strlen(addr))
                  sport = TString(addr) + ":" + sport;
            }

            if (GetServer() && url.HasOption("cors")) {
               const char *cors = url.GetValueFromOptions("cors");
               GetServer()->SetCors(cors && *cors ? cors : "*");
            }

            if (GetServer() && url.HasOption("cred_cors")) {
               const char *cred = url.GetValueFromOptions("cred_cors");
               GetServer()->SetCorsCredentials(cred && *cred ? cred : "true");
            }

            if (url.HasOption("nocache"))
               fMaxAge = 0;

            if (url.HasOption("max_age"))
               fMaxAge = url.GetIntValueFromOptions("max_age");

            max_age.Form("%d", fMaxAge);
         }
      }
   }

   num_threads.Form("%d", fNumThreads);

   const char *options[34];
   int op = 0;

   Info("Create", "Starting HTTP server on port %s", sport.Data());

   options[op++] = "listening_ports";
   options[op++] = sport.Data();
   options[op++] = "num_threads";
   options[op++] = num_threads.Data();

   if (use_ws) {
      options[op++] = "websocket_timeout_ms";
      options[op++] = websocket_timeout.Data();
   }

   if ((auth_file.Length() > 0) && (auth_domain.Length() > 0)) {
      options[op++] = "global_auth_file";
      options[op++] = auth_file.Data();
      options[op++] = "authentication_domain";
      options[op++] = auth_domain.Data();
   } else {
      options[op++] = "enable_auth_domain_check";
      options[op++] = "no";
   }

   if (log_file.Length() > 0) {
      options[op++] = "error_log_file";
      options[op++] = log_file.Data();
   }

   if (ssl_cert.Length() > 0) {
      options[op++] = "ssl_certificate";
      options[op++] = ssl_cert.Data();
   } else if (IsSecured()) {
      Error("Create", "No SSL certificate file configured");
   }

   if (max_age.Length() > 0) {
      options[op++] = "static_file_max_age";
      options[op++] = max_age.Data();
   }

   if (GetServer() && GetServer()->IsCors()) {
      // also used for the file transfer
      options[op++] = "access_control_allow_origin";
      options[op++] = GetServer()->GetCors();
   }

   if (GetServer() && GetServer()->IsCorsCredentials()) {
      options[op++] = "access_control_allow_credentials";
      options[op++] = GetServer()->GetCorsCredentials();
      options[op++] = "access_control_expose_headers";
      options[op++] = "Content-Range, Content-Length, Date";
      options[op++] = "access_control_allow_methods";
      options[op++] = "GET, HEAD, OPTIONS";
   }

   options[op++] = "enable_directory_listing";
   options[op++] = dir_listening.Data();

   options[op++] = nullptr;

   // try to remove socket file - if any
   if (is_socket && !sport.Contains(","))
      gSystem->Unlink(sport.Data()+1);

   // Start the web server.
   fCtx = mg_start(&fCallbacks, this, options);

   if (!fCtx)
      return kFALSE;

   mg_set_request_handler(fCtx, "/", begin_request_handler, nullptr);

   if (use_ws)
      mg_set_websocket_handler(fCtx, "**root.websocket$", websocket_connect_handler,
                               websocket_ready_handler, websocket_data_handler, websocket_close_handler, nullptr);

   // try to remove socket file - if any
   if (is_socket && !sport.Contains(","))
      gSystem->Chmod(sport.Data()+1, socket_mode);

   return kTRUE;
}
