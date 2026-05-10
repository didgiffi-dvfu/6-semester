<?php
require_once('db.php');

if (isset($_COOKIE['User'])) {
    header('Location: /profile.php');
    exit();
}

$error = '';
$login = '';

if (isset($_POST['submit'])) {
    $link = mysqli_connect($servername, $username, $password, $dbName);

    if (!$link) {
        die('Ошибка подключения: ' . mysqli_connect_error());
    }

    mysqli_set_charset($link, 'utf8mb4');

    $login = trim($_POST['login'] ?? '');
    $pass = trim($_POST['password'] ?? '');

    if (!$login || !$pass) {
        $error = 'input all parameters';
    } else {
        $loginSql = mysqli_real_escape_string($link, $login);
        $passSql = mysqli_real_escape_string($link, $pass);

        $sql = "SELECT * FROM users WHERE username='$loginSql' AND password='$passSql'";
        $result = mysqli_query($link, $sql);

        if ($result && mysqli_num_rows($result) == 1) {
            $user = mysqli_fetch_array($result);
            setcookie('User', $user['username'], time() + 7200, '/');
            mysqli_close($link);
            header('Location: /profile.php');
            exit();
        } else {
            $error = 'не правильное имя или пароль';
        }
    }

    mysqli_close($link);
}
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Избосарова Ж.О. - вход</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>

<body class="auth-page">

    <div class="pony-card">

        <img src="Gemini_Generated_Image_2gm88z2gm88z2gm8.png" class="pony-avatar" alt="Аватар">

        <h2>С возвращением ✨</h2>

        <p class="text-muted mb-4">
            Войди в свой уютный магический мир
        </p>

        <?php if ($error !== ''): ?>
            <div class="alert alert-danger py-2" role="alert">
                <?php echo htmlspecialchars($error, ENT_QUOTES, 'UTF-8'); ?>
            </div>
        <?php endif; ?>

        <form action="/login.php" method="POST" class="d-flex flex-column gap-3">

            <input
                type="text"
                name="login"
                class="form-control"
                placeholder="Логин"
                value="<?php echo htmlspecialchars($login, ENT_QUOTES, 'UTF-8'); ?>"
                required
            >

            <input
                type="password"
                name="password"
                class="form-control"
                placeholder="Пароль"
                required
            >

            <button type="submit" name="submit" class="btn btn-pink w-100 mt-2">
                Продолжить ✨
            </button>

        </form>

        <p class="mt-3">
            Нет аккаунта?
            <a href="/registration.php" class="magic-link">Создать</a>
        </p>

    </div>

</body>
</html>
