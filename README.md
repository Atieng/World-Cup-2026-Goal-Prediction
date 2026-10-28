# La Beauté — Backend

A small Node/Express backend for the booking system. Stores bookings in a local
SQLite file (`la-beaute.db`, created automatically), emails a confirmation to
the client and the salon, and gives you a simple admin page to manage bookings.

## 1. Install

```bash
cd backend
npm install
```

## 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set:
- `ADMIN_KEY` — the password for the admin dashboard. Change it from the default.
- `SALON_EMAIL` — where new-booking notifications should go.
- `SMTP_*` — your email provider's SMTP details, so confirmation emails actually
  send. If you leave these blank, bookings still work — the server just prints
  the email to the console instead of sending it, which is fine while testing.

  Easiest option for a Gmail account: turn on 2-Step Verification, then create
  an **App Password** at https://myaccount.google.com/apppasswords and use
  that as `SMTP_PASS` (not your normal Gmail password).

## 3. Run

```bash
npm start
```

You should see:
```
La Beauté backend running at http://localhost:3001
Admin dashboard at        http://localhost:3001/admin
```

Leave this running in its own terminal window.

## 4. Point the website at it

The frontend (`script.js`) already calls `http://localhost:3001` by default
— look for the `API_BASE` constant near the top of the file. Just open
`index.html` with Live Server as before; as long as this backend is running,
"Confirm Reservation" in the booking flow will save to the database and send
the emails.

## 5. Manage bookings

Go to **http://localhost:3001/admin** in your browser, enter your `ADMIN_KEY`,
and you'll see every booking with the ability to mark it Completed / Cancelled
or delete it.

## API reference

| Method | Path                          | Auth        | Purpose                          |
|--------|-------------------------------|-------------|-----------------------------------|
| POST   | `/api/bookings`                | —           | Create a booking (used by the site) |
| GET    | `/api/bookings/code/:code`     | —           | Look up a single booking by its code |
| GET    | `/api/admin/bookings`          | `x-admin-key` header | List every booking        |
| PATCH  | `/api/admin/bookings/:id`      | `x-admin-key` header | Update a booking's status (`confirmed`/`completed`/`cancelled`) |
| DELETE | `/api/admin/bookings/:id`      | `x-admin-key` header | Delete a booking          |
| GET    | `/api/health`                  | —           | Health check                      |

## Notes / next steps

- **SMS confirmations**: not wired up yet. The cleanest way to add this is
  Twilio — sign up, install `twilio` (`npm install twilio`), and call it
  alongside `sendBookingEmails()` in `server.js`. Ask me and I can wire this
  in once you have a Twilio account.
- **Deploying**: this is set up for local testing only right now (SQLite file
  + `localhost`). When you're ready to put this on the internet, the easiest
  path is a small host like Render or Railway, swapping SQLite for a hosted
  Postgres database — happy to help with that when you get there.
- **Double-booking**: the backend doesn't currently block two clients picking
  the same time slot — the frontend just shows a few slots as "taken" for
  demo purposes. Worth adding a real availability check before this goes live.
